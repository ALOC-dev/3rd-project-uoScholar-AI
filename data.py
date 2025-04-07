import mysql.connector

# MySQL 연결
def get_connection():
    return mysql.connector.connect(
        host="uoscholar.cdkke4m4o6zb.ap-northeast-2.rds.amazonaws.com",
        user="admin",
        password="",
        database="uoscholar_db",
        port=3306
    )

# DB 연결하고 테이블 내용 출력
def show_data():
    conn = get_connection()
    cursor = conn.cursor()

    # 조회할 테이블 이름 넣기 (예: posts)
    query = "SELECT * FROM notice;"
    cursor.execute(query)

    # 결과 가져오기
    rows = cursor.fetchall()

    # 컬럼 이름 출력
    column_names = [desc[0] for desc in cursor.description]
    print(" | ".join(column_names))
    print("-" * 50)

    # 데이터 출력
    for row in rows:
        print(" | ".join(str(col) for col in row))

    cursor.close()
    conn.close()

# 실행
show_data()
