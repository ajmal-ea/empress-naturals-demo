FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy frontend files
COPY frontend/static/ ./static/
COPY frontend/test-embed.html .
COPY frontend/dev-server.js .

# Set environment variables
ENV NODE_ENV=production
ENV PORT=10000

# Expose port (using PORT env var)
EXPOSE ${PORT}

# Start the server
CMD ["node", "dev-server.js"] 