FROM node:22-slim

WORKDIR /app

COPY package*.json ./
RUN npm ci --omit=dev

COPY src/ ./src/

ENV GATEWAY_PORT=8000
ENV STAT_URL=http://stat:8002
ENV NLP_URL=http://nlp:8001

EXPOSE 8000

CMD ["node", "src/index.js"]
