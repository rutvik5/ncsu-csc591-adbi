from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def main():
    conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec
    ssc.checkpoint("checkpoint")
    
    sc.setLogLevel('WARN')
    pwords = load_wordlist("positive.txt")
    nwords = load_wordlist("negative.txt")
   
    counts = stream(ssc, pwords, nwords, 100)
    make_plot(counts)


def make_plot(counts):
    
    #Plot the counts for the positive and negative words for each timestep.
    #Use plt.show() so that the plot will popup.
    graph = plt.figure()
    positives=[]
    negatives=[]
    for tup_list in counts:
      for tup in tup_list:
        if 'positive' in tup:
          positives.append(tup[1])
        if 'negative'in tup:
          negatives.append(tup[1])


    plt.plot(positives, '-b', marker='.', label='positive')
    plt.plot(negatives, '-g', marker='.', label='negative') 
    plt.legend(loc='upper left')
    plt.ylabel('Word count')
    plt.xlabel('Time step')
    graph.savefig('plot.png')

def load_wordlist(filename):
    """ 
    This function should return a list or set of words from the given filename.
    """
    with open(filename,'r') as f:
      word_list = f.read().splitlines()
    return set(word_list)



def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1])

    # Each element of tweets will be the text of a tweet.
    # You need to find the count of all the positive and negative words in these tweets.
    # Keep track of a running total counts and print this at every time step (use the pprint function).
    # YOUR CODE HERE
    def categorize_words(word):
        if word in pwords:
          return ('positive', 1)
        elif word in nwords:
          return ('negative', 1)
        else:
          return ('positive', 0)
    
    def update_function(new_count, prev_count):
        return sum(new_count, prev_count or 0)

    tweet_words = tweets.flatMap(lambda x: x.split(' '))
    pairs = tweet_words.map(categorize_words)
    word_counts = pairs.reduceByKey(lambda x,y: x + y)
    total_counts = word_counts.updateStateByKey(update_function)
    total_counts.pprint()
    # Let the counts variable hold the word counts for all time steps
    # You will need to use the foreachRDD function.
    # For our implementation, counts looked like:
    #   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
    counts = []
    word_counts.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))

    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)

    return counts


if __name__=="__main__":
    main()
