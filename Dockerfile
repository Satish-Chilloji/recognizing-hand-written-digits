# Use the previously built Dependency Image as the base
FROM base:latest

# Set the working directory
WORKDIR /digit

# Copy the entire application code into the container
COPY . .

# Run unit tests using pytest
CMD ["pytest"]