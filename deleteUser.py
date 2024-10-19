import pandas as pd
df=pd.read_csv("C:\\Users\\Asus\\Desktop\\BTL\\User.csv")
def delete_user_by_id(user_id):
    global df
    if user_id in df["id"].values:
        df=df[df["id"]!=user_id]

        df.to_csv("C:\\Users\\Asus\\Desktop\\BTL\\User.csv",index=False)
        print("User deleted successfully")
    else:
        print("User not found")
delete_user_by_id(7)