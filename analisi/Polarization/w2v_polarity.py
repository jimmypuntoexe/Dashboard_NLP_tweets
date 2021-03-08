from scipy.spatial.distance import cosine

def w2vPolarity(model, topic, positive, negative):
  positive_score = 0
  negative_score = 0

  for t in topic:

    positive_temp = 0
    for p in positive:
      positive_temp = positive_temp + ( 1 - cosine(model[t], model[p]))
    positive_temp = positive_temp/len(positive)
    

    negative_temp = 0
    for n in negative:
      negative_temp = negative_temp + (1 - cosine(model[t], model[n]))
    negative_temp = negative_temp/len(negative)

    positive_score = positive_score + positive_temp
    negative_score = negative_score + negative_temp

    return   positive_score - negative_score