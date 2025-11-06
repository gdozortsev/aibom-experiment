# Use the Python slim image as requested
FROM python:3.9-slim-bullseye

# Set the application directory
WORKDIR /app

# Copy the application code and requirements
COPY . /app

# --- OPTIMIZED INSTALLATION AND DATA DOWNLOAD (Single RUN layer) ---
# Install build tools (curl, tar, gzip, etc.) necessary for the next steps.
# We must install both curl and the tar/gzip utilities (like gzip or xz-utils) 
# and then use them immediately.
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends \
#         curl \
#         tar \
#         gzip \
#         xz-utils \
#     && rm -rf /var/lib/apt/lists/* \
#     \
#     # Install Python dependencies (must be in the same RUN for efficiency)
#     && pip3 install --no-cache-dir -r requirements.txt \
#     \
#     # Download and extract the datasets
#     # Combine downloads for efficiency
#     && curl -L -O https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz \
#     && curl -L -O https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz \
#     \
#     # Extract images and annotations
#     && tar -xzf images.tar.gz \
#     && tar -xzf annotations.tar.gz \
#     \
#     # Clean up the downloaded tar archives to keep the final image small
#     && rm images.tar.gz annotations.tar.gz

# # Application port exposure
EXPOSE 8080

# # Environment variables
# ENV PYTHONUNBUFFERED=1

# # Command to run the application
# CMD ["python3", "application/app.py"]

#test command
CMD ["echo", "hello world"]