#!/bin/bash
# deploy.sh — 生产部署脚本
#
# 用法: ./deploy.sh [dev|prod]

set -e

ENV=${1:-prod}
echo "🚀 部署环境: $ENV"

# 检查依赖
check_dependency() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ 缺少依赖: $1"
        exit 1
    fi
}

check_dependency docker
check_dependency docker-compose

# 创建必要目录
echo "📁 创建目录..."
mkdir -p data
mkdir -p ssl
mkdir -p logs

# 复制配置文件
echo "📋 复制配置..."
if [ ! -f settings.json ]; then
    echo "⚠️ 警告: settings.json 不存在，创建默认配置"
    cat > settings.json << 'EOF'
{
  "default_vendor": "deepseek",
  "default_model": "deepseek-chat",
  "vendor_creds": {},
  "system_prompt": "You are a helpful AI assistant.",
  "temperature": 0.7,
  "max_tokens": 4096
}
EOF
fi

# 构建镜像
echo "🔨 构建 Docker 镜像..."
docker-compose build

# 停止旧服务
echo "🛑 停止旧服务..."
docker-compose down --remove-orphans

# 启动新服务
echo "▶️ 启动服务..."
if [ "$ENV" = "dev" ]; then
    docker-compose up
else
    docker-compose up -d
fi

# 等待服务就绪
echo "⏳ 等待服务就绪..."
sleep 5

# 健康检查
echo "🏥 健康检查..."
if curl -sf http://localhost:5000/health > /dev/null; then
    echo "✅ 服务运行正常"
else
    echo "❌ 健康检查失败"
    docker-compose logs --tail=50
    exit 1
fi

echo "🎉 部署完成!"
echo ""
echo "访问地址:"
echo "  - 应用: http://localhost:5000"
echo "  - 健康检查: http://localhost:5000/health"
echo ""
echo "常用命令:"
echo "  查看日志: docker-compose logs -f"
echo "  停止服务: docker-compose down"
echo "  重启服务: docker-compose restart"
