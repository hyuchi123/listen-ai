FROM golang:1.24-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY main.go .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o stat .

FROM alpine:3.20

WORKDIR /app

COPY --from=builder /app/stat .

ENV STAT_PORT=8002
ENV SQLITE_PATH=/data/listenai.db

EXPOSE 8002

CMD ["./stat"]
