import jwt
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJBMDAxIiwiZnVsbF9uYW1lIjoiXHU3Y2ZiXHU3ZDcxXHU3YmExXHU3NDA2XHU1NGUxIiwicm9sZSI6ImFkbWluIiwiZXhwIjoxNzg2MDAxNDg2fQ.I2Ljn-pupOzwUjORZsEiDOTrvcRizrAg4hkaACXLZbg"
secret = "afb5dee5f4a10779c824a35cab8955863da98ec3eee0c3c43c6e3b3f28a4e413"
try:
    decoded = jwt.decode(token, secret, algorithms=["HS256"])
    print("Success:", decoded)
except Exception as e:
    print("Error:", e)
