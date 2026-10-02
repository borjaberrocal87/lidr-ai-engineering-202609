# syntax=docker/dockerfile:1
FROM node:22-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY src ./src
COPY views ./views
COPY public ./public
COPY migrations ./migrations

ENV NODE_ENV=production
EXPOSE 3000

CMD ["node", "src/server.js"]
