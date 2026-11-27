# check=skip=SecretsUsedInArgOrEnv
# Ariadne 网页版镜像：前端构建产物由 FastAPI 一起托管，一个容器、一个端口（8000）。
#
#   docker build -t ariadne .
#   docker run -d -p 8000:8000 -v ariadne-data:/data -e ARIA_PASSWORD=换成你的密码 -e LLM_API_KEY=sk-... ariadne
#
# 镜像默认要求设置访问密码（ARIA_PASSWORD），否则拒绝启动；只在本机自用可加 -e ARIA_REQUIRE_PASSWORD=0
#
# 国内网络：基础镜像、npm、PyPI 都可以换成镜像源（docker compose 用户在 deploy/ariadne.env 旁边的 .env 里设置，见 README）
#   docker build --build-arg NODE_IMAGE=<镜像站>/library/node:22-slim \
#                --build-arg PYTHON_IMAGE=<镜像站>/library/python:3.12-slim \
#                --build-arg NPM_REGISTRY=https://registry.npmmirror.com \
#                --build-arg PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple -t ariadne .
#
# 注意：不要加 "# syntax=docker/dockerfile:1"，它会在构建前去 Docker Hub 下载解析器，国内常常连不上；
# 下面用到的 RUN --mount 等语法 Docker 23 及以上内置支持。

ARG NODE_IMAGE=node:22-slim
ARG PYTHON_IMAGE=python:3.12-slim

# ---------------------------------------------------------------- 前端
FROM ${NODE_IMAGE} AS web
ARG NPM_REGISTRY=https://registry.npmjs.org
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --no-audit --no-fund --registry="$NPM_REGISTRY"
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------- 后端
FROM ${PYTHON_IMAGE}
ARG PYPI_INDEX=https://pypi.org/simple
# uv 从 PyPI 安装（而不是 ghcr.io 上的 uv 镜像），这样换了 PyPI 镜像源就不再需要访问 ghcr.io
RUN python -m pip install --no-cache-dir --disable-pip-version-check --index-url "$PYPI_INDEX" "uv==0.11.32"

ENV UV_DEFAULT_INDEX=${PYPI_INDEX} \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    ARIA_DATA_DIR=/data \
    ARIA_FRONTEND_DIST=/app/frontend/dist \
    ARIA_REQUIRE_PASSWORD=1

WORKDIR /app/backend
# 先只装依赖，代码改动时可以复用这一层缓存
COPY backend/pyproject.toml ./
# The backend's assessment pipeline shares only this stdlib data package.
# Training code, experiment environments, corpora and weights stay outside.
COPY experiments/common/pyproject.toml experiments/common/README.md /app/experiments/common/
COPY experiments/common/ariadne_data/ /app/experiments/common/ariadne_data/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project

COPY backend/app ./app
COPY --from=web /src/frontend/dist /app/frontend/dist

# 用普通用户运行；文献库、论文 PDF、设置都在 /data（挂载卷保存）
RUN useradd --create-home --uid 10001 aria \
    && mkdir -p /data && chown aria:aria /data
USER aria
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
