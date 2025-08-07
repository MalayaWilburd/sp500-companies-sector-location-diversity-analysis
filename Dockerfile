# Use a lightweight Python image as the base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Create a .streamlit directory with full permissions
RUN mkdir -p /app/.streamlit
RUN chmod 777 /app/.streamlit

# Copy the requirements file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the rest of your project files into the container
COPY . .

# Set the Streamlit configuration directory
ENV STREAMLIT_CONFIG_FOLDER="/app/.streamlit"

# Expose the default port for Streamlit
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]