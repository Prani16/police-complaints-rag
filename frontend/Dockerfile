# Frontend image: Node.js Express web UI + CLI.
#   docker build -t rag-frontend frontend/
FROM node:22-slim
ENV NODE_ENV=production
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev
COPY backend-client.js server.js cli.js ./
COPY public ./public
USER node
EXPOSE 8080
ENV PORT=8080
CMD ["node", "server.js"]
