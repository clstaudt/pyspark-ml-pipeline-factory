from pyspark.sql import functions as F

from pyspark.ml import Pipeline, Transformer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, OneHotEncoderEstimator, StringIndexer

from pyspark.mllib.evaluation import MulticlassMetrics

import pandas


# custom transformers as pipeline components

class CustomTransformer(Transformer):
    # lazy workaround - a transformer needs to have these attributes
    _defaultParamMap = dict()
    _paramMap = dict()
    _params = dict()
    uid = 0
    
    def __repr__(self):
        return str(type(self))

class ColumnSelector(CustomTransformer):
    """Transformer that selects a subset of columns
    - to be used as pipeline stage"""

    def __init__(self, columns):
        self.columns = columns


    def _transform(self, data):
        return data.select(self.columns)
    



class ColumnRenamer(CustomTransformer):
    """Transformer renames one column"""

    def __init__(self, rename):
        self.rename = rename

    def _transform(self, data):
        (colNameBefore, colNameAfter) = self.rename
        return data.withColumnRenamed(colNameBefore, colNameAfter)

    
class ColumnDropper(CustomTransformer):
    """Transformer drops list of columns"""

    def __init__(self, colNames):
        self.colNames = colNames

    def _transform(self, data):
        for colName in self.colNames:
            print("dropping ", colName)
            data = data.drop(data.col(colName))
        return data

    def __repr__(self):
        return "ColumnDropper: " + ",".join(self.colNames)
    
    
class NaDropper(CustomTransformer):
    """
    Drops rows with at least one not-a-number element
    """

    def __init__(self, cols=None):
        self.cols = cols


    def _transform(self, data):
        dataAfterDrop = data.dropna(subset=self.cols) 
        return dataAfterDrop


class ColumnCaster(CustomTransformer):

    def __init__(self, col, toType):
        self.col = col
        self.toType = toType

    def _transform(self, data):
        print("casting column {0} to {1}".format(self.col, self.toType))
        return data.withColumn(self.col, data[self.col].cast(self.toType))


class MLPipelineFactory:

    # dict: algorithm_name -> (algorith_class, params)
    classifiers = {
        "Decision Tree": (DecisionTreeClassifier, dict()),
        "Random Forest": (RandomForestClassifier, dict()),
        "Logistic Regression": (LogisticRegression, dict()),
    }

    regressors = {
        "Boosted Trees": (GBTRegressor, dict()),
    }


    def __init__(self, problemType, data, target, numericCols, categoricalCols, categoricalEncoding="index", dropna=True, algorithm=None):
        """ """
        self.problemType = problemType
        self.data = data  # TODO: copy dataframe?
        self.algorithm = algorithm
        self.target = target
        self.numericCols = numericCols
        self.categoricalCols = categoricalCols
        self.categoricalEncoding = categoricalEncoding
        self.algorithm = algorithm
        self.dropna = True  # drop rows with missing elements before assembling feature vector
        self.pipelines = dict() # dictionary of different pipelines assembled
        # select default algorithm
        if algorithm is None:
            if problemType is "classification":
                self.algorithm = "Random Forest"
            elif problemType is "regression":
                self.algorithm = "Boosted Trees"


    def make(self):
        """Run pipeline, train"""

        preproStages = [] # stages of preprocessing pipeline
        inputCols = []  # list of columns that will be assembled into a feature vector
        # numeric columns can be used as input without encoding
        inputCols = self.numericCols
        
        # encode categorical columns
        if self.categoricalCols is not None:
            if self.categoricalEncoding is "index":
                handleCategoricals = [StringIndexer(inputCol=col, outputCol="{0}_indexed".format(col)).setHandleInvalid("skip")
                for col in self.categoricalCols]
                inputCols += ["{0}_indexed".format(col) for col in self.categoricalCols]
            elif self.categoricalEncoding is "onehot":
                # OneHotEncoderEstimator expects indexed categories
                handleCategoricals = [StringIndexer(inputCol=col, 
                                                   outputCol="{0}_indexed".format(col)).setHandleInvalid("skip")
                for col in self.categoricalCols]
                handleCategoricals += [OneHotEncoderEstimator(inputCols=["{0}_indexed".format(col) for col in self.categoricalCols], 
                                                              outputCols=[col + "_1h" for col in self.categoricalCols])]
                inputCols += ["{0}_1h".format(col) for col in self.categoricalCols]
            else:
                raise Exception("unknown categorical encoding: ", self.categoricalEncoding)
            preproStages +=  handleCategoricals
        # drop na rows
        if self.dropna:
            preproStages.append(NaDropper(inputCols))
        # assemble feature vectors
        print("assembling feature vectors, using input columns: ", inputCols)
        self.inputCols = inputCols # store for later access
        assembleFeatures = VectorAssembler(inputCols=inputCols, outputCol="features")
        preproStages.append(assembleFeatures)

        # if target column is boolean, cast to int as expecte by pyspark.ml
        if self.data.select(self.target).dtypes[0][1] == "boolean":
            preproStages.append(ColumnCaster(self.target, "int"))

        # select feature and target column only
        selectFeaturesAndTarget = ColumnSelector(["features", self.target])
        preproStages.append(selectFeaturesAndTarget)

        finalStages = []
        # rename target column to 'label' as expected by spark.ml
        finalStages.append(ColumnRenamer((self.target, "label")))

        if self.problemType is "regression":
            (learnerClass, learnerParams) = self.regressors[self.algorithm]
        elif self.problemType is "classification":
            (learnerClass, learnerParams)  = self.classifiers[self.algorithm]
        else:
            raise Exception("unknown prediction type", self.problemType)
        learner = learnerClass(**learnerParams)
        finalStages.append(learner)
        # cast predicted label from double to int if classification problem
        #if self.problemType is "classification":
        #    finalStages.append(ColumnCaster("prediction", "int"))

        # assemble
        self.pipelines["pre"] = Pipeline(stages=preproStages)
        self.pipelines["post"] = Pipeline(stages=finalStages)
        self.pipelines["complete"] = Pipeline(stages=preproStages + finalStages)
    
    
    def getPipeline(self, part="complete"):
        return self.pipelines[part]
    
    def getInputColumns(self):
        return self.inputCols
        


def evaluateClassifier(pipelineFactory, data):
    splitRatio=0.8
    training, test = data.randomSplit([splitRatio, 1-splitRatio])
    training = training.cache()

    #pipelineFactory.make()
    pipeline = pipelineFactory.getPipeline()
    # fit model to training set
    model = pipeline.fit(training)
    # predict on test set
    predictions = model.transform(test)
    
    # MulticlassMetrics expects label to be of type double
    predictions = (predictions.withColumn("label", predictions["label"].cast("double")))
    
    mm = MulticlassMetrics(predictions.select(["label", "prediction"]).rdd)
    labels = sorted(predictions.select("prediction").rdd.distinct().map(lambda r: r[0]).collect())
    
    metrics = pandas.DataFrame([(label, mm.precision(label=label), mm.recall(label=label), mm.fMeasure(label=label)) for label in labels],
                            columns=["label", "Precision", "Recall", "F1"])
    # TODO: verify
    return metrics


def computeFeatureImportances(pipelineFactory, data):
    # TODO: fit model to entire data set?
    #pipelineFactory.make()
    pipeline = pipelineFactory.getPipeline()
    cols = pipelineFactory.getInputColumns()
    model = pipeline.fit(data)
    predictor = model.stages[-1]
    importances = pandas.Series(predictor.featureImportances, index=cols)
    importances.sort_values(ascending=False, inplace=True)
    return importances