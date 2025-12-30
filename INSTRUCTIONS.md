## Quick Start Instructions


### Prerequisites

- You only need **Docker Desktop** installed on your computer.
---

### Step 1: Extract the Project

- Unzip the `qa-gen.zip` file to any folder on your computer.

---

## Step 3: Run the Application
- Make Sure Docker is Running
- Open a terminal/command prompt in the extracted folder and run:

```bash
sudo docker-compose up --build
```

**Wait for this message:**
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Access the Application

- Open your browser and go to:

  ```
  http://localhost:8000/docs
  ```

You'll see the interactive API interface where you can:
1. Upload PDFs
2. Generate questions

---

## Quick Test

1. Go to `http://localhost:8000/docs`
2. Click **POST /api/ingest** → **Try it out**
3. Upload a PDF file
4. Click **Execute**
5. Copy the `filename` from the response
6. Click **POST /api/generate/questions** → **Try it out**
7. Enter:
   ```json
   {
     "filename": "your-file.pdf",
     "query": "main concepts",
     "num_questions": 3
   }
   ```
8. Click **Execute**
9. See your generated questions!


---

##  Troubleshooting

- **Cannot connect to Docker daemon**
→ Make sure Docker Desktop is running

- **Port 8000 already in use**
→ Stop any other application using port 8000, or change the port in `docker-compose.yml`

- **Application is slow**
→ First request takes longer. Subsequent requests are faster.

---

