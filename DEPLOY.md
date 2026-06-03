# 部署指南

## 快速开始

### 使用 Docker Compose（推荐）

```bash
# 1. 克隆项目
git clone <repo-url>
cd llama-cpp_vlm_web

# 2. 配置
# 编辑 settings.json 添加 API 密钥

# 3. 部署
./deploy.sh prod

# 4. 访问
# 打开浏览器访问 http://localhost:5000
```

### 手动部署

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置
# 编辑 settings.json

# 3. 使用 Gunicorn 启动
gunicorn -c gunicorn.conf.py app:app
```

## 配置说明

### settings.json

```json
{
  "default_vendor": "deepseek",
  "default_model": "deepseek-chat",
  "vendor_creds": {
    "deepseek": {
      "api_key": "your-api-key",
      "base_url": "https://api.deepseek.com/v1"
    }
  }
}
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `FLASK_ENV` | 运行环境 | `production` |
| `PORT` | 监听端口 | `5000` |
| `WORKERS` | Gunicorn 工作进程数 | `CPU * 2 + 1` |

## Nginx 配置

```bash
# 复制配置
cp nginx.conf /etc/nginx/nginx.conf

# 测试配置
nginx -t

# 重载配置
nginx -s reload
```

## SSL 证书

使用 Let's Encrypt：

```bash
# 安装 certbot
certbot certonly --webroot -w /var/www/certbot -d your-domain.com

# 复制证书
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ssl/cert.pem
cp /etc/letsencrypt/live/your-domain.com/privkey.pem ssl/key.pem
```

## 监控

### 健康检查

```bash
curl http://localhost:5000/health
```

### 性能指标

```bash
curl http://localhost:5000/metrics
```

## 故障排查

### 查看日志

```bash
# Docker
docker-compose logs -f

# 手动
tail -f logs/gunicorn.log
```

### 重启服务

```bash
# Docker
docker-compose restart

# 手动
pkill gunicorn
gunicorn -c gunicorn.conf.py app:app
```

## 更新

```bash
# 拉取最新代码
git pull

# 重新构建
docker-compose build

# 重启
docker-compose up -d
```
