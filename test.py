import pandas as pd
df = pd.read_csv("data/library_clean.csv")
print(df.shape)
print(df[df["Artist"].eq("iPhone de Henrique")][["Name","Artist","Genre"]])


from genre_classifier import APIFetcher
api = APIFetcher()
print(api.fetch("MC Kevinho"))
