
import psycopg2
import face_recognition

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    database="face_rec",
    user="root",
    password="1",
    host="127.0.0.1",
    port="5432"
)

# Create a cursor object to interact with the database
cursor = conn.cursor()

# Load an image and compute the face encoding
image_path = "./face/front.jpg"
image = face_recognition.load_image_file(image_path)
face_encoding = face_recognition.face_encodings(image)

# Assuming you have a table named 'face_encodings' with columns 'person_id' and 'encoding'
# You may need to create this table in your database before running the script

# Insert the face encoding into the database
person_id = 1  # You can assign a unique ID for each person
cursor.execute("INSERT INTO face_encodings (person_id, encoding) VALUES (%s, %s)", (person_id, face_encoding[0].tobytes()))

# Commit the changes
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()
