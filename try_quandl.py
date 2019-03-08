import quandl
import pandas

# my API Key: 4dBpomttZwaktu1hAdDB

x = quandl.get("CHRIS/CME_ES1", authtoken="4dBpomttZwaktu1hAdDB")
print(x)
print(type(x))

