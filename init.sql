-- init.sql
drop TABLE IF NOT EXISTS face_encodings ;
CREATE TABLE IF NOT EXISTS face_encodings (
    user_id SERIAL PRIMARY KEY,
    encoding BYTEA NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
