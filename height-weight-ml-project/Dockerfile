# Use official lightweight Python image
FROM python:3.10-slim


WORKDIR /app


COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


COPY app.py app.py
COPY height_weight.csv height_weight.csv


EXPOSE 8050

# Run the app
CMD ["python", "app.py"]
