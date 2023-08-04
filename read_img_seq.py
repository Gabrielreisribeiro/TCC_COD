from pyspark import SparkConf, SparkContext
path = "INRIA/set00/V000.seq"

conf = SparkConf()
conf.setMaster("local").setAppName("Spark Example App")
sc = SparkContext.getOrCreate(conf)
a = sc.sequenceFile(path = path)
print(a)