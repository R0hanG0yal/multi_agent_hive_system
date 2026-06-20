FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HOST=0.0.0.0 \
    PORT=7860 \
    EPISODES=40 \
    APP_STATE_PATH=/data/hackathon_ai_env_state.json

RUN useradd -m -u 1000 user
RUN mkdir -p /data && chown user:user /data

USER user
WORKDIR $HOME/app

COPY --chown=user . $HOME/app

EXPOSE 7860

CMD ["sh", "-c", "python3 main.py web --host ${HOST:-0.0.0.0} --port ${PORT:-7860} --episodes ${EPISODES:-40}"]
