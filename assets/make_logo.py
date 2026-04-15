"""
# 반드시 이 위치에서 실행
E:\Develop\eva_nexio_daia_v5> python assets/make_logo.py
python assets/make_logo.py

"""



import base64
with open("assets/logo-title.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
with open("assets/logo_b64.py", "w") as f:
    f.write(f'LOGO_B64 = "{b64}"\n')
print("완료")