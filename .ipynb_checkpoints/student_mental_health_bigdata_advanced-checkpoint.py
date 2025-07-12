import os
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import ClusteringEvaluator, BinaryClassificationEvaluator, MulticlassClassificationEvaluator

EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

spark = SparkSession.builder.appName("StudentMentalHealthBigDataAdvanced").getOrCreate()

data_path = "students_mental_health.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)
df.printSchema()
df.show(5)

key_columns = ['Age', 'CGPA', 'Depression_Score', 'Anxiety_Score']
df = df.na.drop(subset=key_columns)

fillna_cols = [
    'Gender', 'Sleep_Quality', 'Physical_Activity', 'Diet_Quality',
    'Social_Support', 'Relationship_Status', 'Substance_Use',
    'Counseling_Service_Use', 'Family_History', 'Chronic_Illness',
    'Extracurricular_Involvement', 'Residence_Type'
]
for c in fillna_cols:
    df = df.fillna({c: "Unknown"})

categorical_cols = fillna_cols + ['Course']
indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid='keep') for c in categorical_cols]
for indexer in indexers:
    df = indexer.fit(df).transform(df)
encoder = OneHotEncoder(inputCols=[c+"_idx" for c in categorical_cols], outputCols=[c+"_onehot" for c in categorical_cols])
df = encoder.fit(df).transform(df)

feature_cols = [
    'Age', 'CGPA', 'Stress_Level', 'Depression_Score', 'Anxiety_Score',
    'Financial_Stress', 'Semester_Credit_Load'
] + [c+"_idx" for c in categorical_cols]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")
df = assembler.transform(df)
scaler = StandardScaler(inputCol="features_vec", outputCol="features", withMean=True, withStd=True)
scalerModel = scaler.fit(df)
df = scalerModel.transform(df)

onehot_features = ['Age', 'CGPA', 'Stress_Level', 'Depression_Score', 'Anxiety_Score', 'Financial_Stress', 'Semester_Credit_Load'] + [c+"_onehot" for c in categorical_cols]
assembler2 = VectorAssembler(inputCols=onehot_features, outputCol="features_onehot")
df = assembler2.transform(df)

df.select(['Age','CGPA','Stress_Level','Depression_Score','Anxiety_Score','Financial_Stress']).summary().write.csv(f"{EXPORT_DIR}/eda_summary_stats.csv", header=True, mode='overwrite')
df.withColumn('Depressed', (col('Depression_Score') >= 7).cast('int')).groupBy('Gender').agg(
    F.count('*').alias('Total'),
    F.sum('Depressed').alias('Depressed_Count'),
    (F.sum('Depressed')/F.count('*')).alias('Depression_Rate')
).write.csv(f"{EXPORT_DIR}/eda_gender_vs_depression.csv", header=True, mode='overwrite')
df.groupBy('Sleep_Quality').agg(F.avg('Anxiety_Score').alias('Mean_Anxiety_Score')).write.csv(f"{EXPORT_DIR}/eda_sleep_quality_anxiety.csv", header=True, mode='overwrite')
df.groupBy('Course').agg(F.avg('Depression_Score').alias('Mean_Depression')).orderBy(F.desc('Mean_Depression')).write.csv(f"{EXPORT_DIR}/eda_depression_by_course.csv", header=True, mode='overwrite')
df.groupBy('Financial_Stress').agg(F.avg('CGPA').alias('Avg_CGPA')).write.csv(f"{EXPORT_DIR}/eda_financialstress_vs_cgpa.csv", header=True, mode='overwrite')

na_cols = [F.when(F.col(c).isNull(), 1).otherwise(0).alias(c+'_na') for c in df.columns]
na_df = df.select(*na_cols)
na_count_df = na_df.select([F.sum(c).alias(c) for c in na_df.columns])
na_count_df.write.csv(f"{EXPORT_DIR}/eda_missing_heatmap.csv", header=True, mode='overwrite')

try:
    import pandas as pd
    num_cols = ['Age', 'CGPA', 'Stress_Level', 'Depression_Score', 'Anxiety_Score', 'Financial_Stress', 'Semester_Credit_Load']
    corr_pd = df.select(num_cols).toPandas().corr()
    corr_pd.to_csv(f"{EXPORT_DIR}/eda_correlation_matrix.csv")
    corr_long = pd.melt(corr_pd.reset_index(), id_vars='index')
    corr_long.columns = ['Feature1', 'Feature2', 'Correlation']
    corr_long.to_csv(f"{EXPORT_DIR}/eda_correlation_long.csv", index=False)
except:
    print("Correlation matrix skipped (pandas error or dataset too large).")

for c in ['Gender', 'Sleep_Quality', 'Diet_Quality', 'Relationship_Status', 'Residence_Type']:
    df.groupBy(c).count().orderBy(F.desc('count')).write.csv(f"{EXPORT_DIR}/eda_count_{c}.csv", header=True, mode='overwrite')

kmeans = KMeans(featuresCol='features', k=4, seed=42)
kmeans_model = kmeans.fit(df)
df = kmeans_model.transform(df)
clustering_evaluator = ClusteringEvaluator(featuresCol='features', predictionCol='prediction', metricName='silhouette')
silhouette = clustering_evaluator.evaluate(df)
print(f"KMeans Silhouette Score: {silhouette}")

df.groupBy('prediction','Gender','Sleep_Quality','Social_Support').count().write.csv(f"{EXPORT_DIR}/eda_cluster_profiles.csv", header=True, mode='overwrite')
for feat in ['CGPA','Stress_Level','Depression_Score','Anxiety_Score','Financial_Stress']:
    df.groupBy('prediction').agg(F.avg(feat).alias(f'Avg_{feat}')).orderBy('prediction').write.csv(f"{EXPORT_DIR}/eda_cluster_mean_{feat}.csv", header=True, mode='overwrite')

pca = PCA(k=2, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(df)
df_pca = pca_model.transform(df)
df_pca.select('Age', 'Gender', 'Depression_Score', 
              F.col('pca_features')[0].alias('PCA1'),
              F.col('pca_features')[1].alias('PCA2')).write.csv(
                  f"{EXPORT_DIR}/viz_pca2d.csv", header=True, mode='overwrite')

df.withColumn('Depressed', (col('Depression_Score') >= 7).cast('int'))     .groupBy('Gender', 'Course')     .agg(F.count('*').alias('Total'),
         F.sum('Depressed').alias('Depressed_Count'),
         (F.sum('Depressed')/F.count('*')).alias('Depression_Rate'))     .write.csv(f"{EXPORT_DIR}/eda_gender_course_depression.csv", header=True, mode='overwrite')
df.groupBy('Residence_Type', 'Sleep_Quality').agg(
    F.avg('Anxiety_Score').alias('Mean_Anxiety_Score')).write.csv(
        f"{EXPORT_DIR}/eda_residence_sleep_anxiety.csv", header=True, mode='overwrite')
if 'prediction' in df.columns:
    df.groupBy('prediction','Gender','Relationship_Status').count().write.csv(
        f"{EXPORT_DIR}/viz_cluster_gender_relationship.csv", header=True, mode='overwrite')

if 'prediction' in df.columns:
    df = df.drop('prediction')

risk_threshold = 7
df = df.withColumn("High_Depression_Risk", when(col("Depression_Score") >= risk_threshold, 1).otherwise(0))
train_df, test_df = df.randomSplit([0.8, 0.2], seed=123)
log_reg = LogisticRegression(featuresCol="features", labelCol="High_Depression_Risk", maxIter=20)
log_model = log_reg.fit(train_df)
preds = log_model.transform(test_df)

rf = RandomForestClassifier(featuresCol="features_onehot", labelCol="High_Depression_Risk", numTrees=50)
rf_model = rf.fit(train_df)
rf_importances = rf_model.featureImportances

oh_cols = []
for c in onehot_features:
    if 'onehot' in c:
        try:
            n = df.select(c).head()[c].size
        except:
            n = 0
        oh_cols.extend([c+f"_{i}" for i in range(n)])
    else:
        oh_cols.append(c)
feature_importance_list = list(zip(oh_cols, rf_importances.toArray()))
import pandas as pd
pd.DataFrame(feature_importance_list, columns=['Feature','Importance']).to_csv(f"{EXPORT_DIR}/rf_feature_importances.csv", index=False)

evaluator_roc = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="High_Depression_Risk", metricName="areaUnderROC")
roc_auc = evaluator_roc.evaluate(preds)
evaluator_acc = MulticlassClassificationEvaluator(labelCol="High_Depression_Risk", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_acc.evaluate(preds)
precision = MulticlassClassificationEvaluator(labelCol="High_Depression_Risk", predictionCol="prediction", metricName="precisionByLabel").evaluate(preds)
recall = MulticlassClassificationEvaluator(labelCol="High_Depression_Risk", predictionCol="prediction", metricName="recallByLabel").evaluate(preds)
f1 = MulticlassClassificationEvaluator(labelCol="High_Depression_Risk", predictionCol="prediction", metricName="f1").evaluate(preds)
print(f"Classification Results - ROC AUC: {roc_auc}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")

metrics_table = pd.DataFrame([
    {'Model': 'Logistic Regression', 'Accuracy': accuracy, 'F1': f1, 'ROC AUC': roc_auc, 'Precision': precision, 'Recall': recall}
])
metrics_table.to_csv(f"{EXPORT_DIR}/viz_model_comparison.csv", index=False)

def export_confusion_matrix(preds, export_path):
    cm = preds.groupBy('High_Depression_Risk','prediction').count().orderBy('High_Depression_Risk','prediction')
    cm.write.csv(export_path, header=True, mode='overwrite')
export_confusion_matrix(preds, f"{EXPORT_DIR}/confusion_matrix.csv")

preds.select('High_Depression_Risk', 'probability', 'prediction').write.csv(
    f"{EXPORT_DIR}/viz_logreg_probabilities.csv", header=True, mode='overwrite')

df_export = df.select(
    'Age', 'Gender', 'CGPA', 'Stress_Level', 'Depression_Score', 'Anxiety_Score', 'Sleep_Quality',
    'Financial_Stress', 'High_Depression_Risk'
)
df_export.write.csv(f"{EXPORT_DIR}/student_mental_health_final_outputs.csv", header=True, mode='overwrite')

spark.stop()
