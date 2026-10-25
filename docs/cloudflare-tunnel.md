# 通过 Cloudflare Tunnel 开放公网访问

Cloudflare Tunnel 将一个固定的公网域名连接到运行 Ariadne 的电脑。访问路径为：

```text
浏览器 → https://agent.tanke.asia → Cloudflare → cloudflared → http://127.0.0.1:8000
```

`npm start` 会构建前端，并由 FastAPI 在 8000 端口统一提供页面和 API。隧道连接到这个端口即可，访问者无需与源电脑处于同一局域网。本指南使用 Windows PowerShell；所有 `.\tools\cloudflared.exe` 命令都从仓库根目录执行。

## 当前体验站配置

| 项目 | 当前值 |
|---|---|
| 体验网址 | <https://agent.tanke.asia/> |
| Cloudflare 域名 | `tanke.asia` |
| 隧道名称 | `agent` |
| 隧道 UUID | `ec20bd57-1895-4522-906f-56e45347f7cf` |
| 本地服务 | `http://127.0.0.1:8000` |
| 本机工具 | `D:\code\Ariadnian-agent\tools\cloudflared.exe` |
| 本机配置 | `C:\Users\tk\.cloudflared\config.yml` |
| 本机凭据路径 | `C:\Users\tk\.cloudflared\ec20bd57-1895-4522-906f-56e45347f7cf.json` |

这组值记录项目当前部署。**其他人自行部署时，请使用自己控制的域名、自己创建的隧道 UUID 和自己电脑上的凭据路径。** 隧道名称可以也叫 `agent`，但在另一个账号创建的同名隧道会有不同的 UUID。

## 准备工具与隧道

如果本机已经有 `agent` 隧道、DNS 和配置文件，直接跳到「运行与日常演示」，不必重复创建。

1. 按 [项目快速开始](../README.md#快速开始) 安装依赖，确认 Ariadne 可以本地运行。
2. 将自己的域名添加到 Cloudflare 并完成域名接入。
3. 从 [Cloudflare 官方下载页](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) 下载适合 Windows 的 `cloudflared`，重命名为 `cloudflared.exe` 并放入项目的 `tools` 目录。该目录已被 Git 忽略，克隆仓库后需要自行准备工具。

以下授权与创建命令只在首次配置自己的隧道时运行：

```powershell
cd D:\code\Ariadnian-agent
.\tools\cloudflared.exe --version
.\tools\cloudflared.exe tunnel login
.\tools\cloudflared.exe tunnel create agent
```

`login` 会打开浏览器，让你登录并选择授权的域名；`create` 会生成隧道 UUID 和本机凭据 JSON。记下输出中的 UUID 与凭据文件位置。项目现有的 `agent` 已创建，不要为日常演示再次执行 `create`。

凭据 JSON、`cert.pem`、隧道 Token、访问密码和模型 Key 都应留在本机，不要提交到仓库或贴进文档。

## 第 1 步：把子域名绑定到隧道

当前项目对应的命令为：

```powershell
cd D:\code\Ariadnian-agent
.\tools\cloudflared.exe tunnel route dns agent agent.tanke.asia
```

命令会在 Cloudflare DNS 中创建指向该隧道的 CNAME 记录。当前体验站已经绑定完成，日常启动无需重复执行。

自行部署时，将 `agent.tanke.asia` 换成自己的地址，例如 `demo.example.com`；子域名可以自选，主域名需要属于你已在 Cloudflare 接入并授权的域名。后面的 `hostname` 也要使用同一个地址。如果提示 DNS 记录已经存在，先检查该记录是否已指向目标隧道，不要直接覆盖正在使用的其他记录。

## 第 2 步：创建配置文件

当前机器在 `C:\Users\tk\.cloudflared\config.yml` 保存以下配置：

```yaml
tunnel: ec20bd57-1895-4522-906f-56e45347f7cf
credentials-file: 'C:\Users\tk\.cloudflared\ec20bd57-1895-4522-906f-56e45347f7cf.json'

ingress:
  - hostname: agent.tanke.asia
    service: http://127.0.0.1:8000
  - service: http_status:404
```

`tunnel` 是隧道 UUID；`credentials-file` 指向创建隧道时生成的本机 JSON 文件；`hostname` 是对外域名；`service` 是 `cloudflared` 能访问到的本地应用地址。最后的 `http_status:404` 是必需的兜底规则：不匹配前面域名的请求返回 404。

其他电脑的默认配置目录为 `$env:USERPROFILE\.cloudflared`，请替换 YAML 中的 UUID、完整凭据路径和域名。不要照搬当前部署的 UUID 或假设自己电脑上存在这个 JSON。若配置文件已有其他隧道或路由，应先检查内容，再决定使用独立配置文件，避免覆盖已有配置。

这里假设 `cloudflared` 和 Ariadne 在同一台电脑上。`npm start` 默认监听 `127.0.0.1:8000`，无需为了同机隧道改成 `0.0.0.0`，也无需把本机 8000 端口直接开放到公网。

## 第 3 步：配置访问密码并运行

对外提供真实模型功能前，在 `backend/.env` 中设置以下内容，并将示例密码替换为自己的强密码：

```dotenv
ARIA_PASSWORD=替换为你的强密码
ARIA_LOCK_SETTINGS=1
```

`ARIA_PASSWORD` 开启网页访问密码；未指定 `ARIA_USERNAME` 时，登录框的用户名可以任意填写。`ARIA_LOCK_SETTINGS=1` 禁止访客修改模型和密钥配置，也禁止测试任意模型地址，但**不会阻止已获准访问的用户调用模型**。模型调用仍会产生费用。

可以另外设置 `ARIA_REQUIRE_PASSWORD=1`，让未设置访问密码的应用拒绝启动。修改这些配置后，需要重启 `npm start`。Docker 部署的对应配置写在 `deploy/ariadne.env`，而不是本机启动所用的 `backend/.env`。

打开两个 PowerShell 终端，每个终端都先进入项目目录：

```powershell
# 终端 1：启动 Ariadne；如果应用已正常运行，无需重复启动
cd D:\code\Ariadnian-agent
npm start
```

```powershell
# 终端 2：运行已有隧道
cd D:\code\Ariadnian-agent
.\tools\cloudflared.exe tunnel run agent
```

若配置放在其他位置，显式指定路径：

```powershell
cd D:\code\Ariadnian-agent
.\tools\cloudflared.exe tunnel --config 'C:\Users\tk\.cloudflared\config.yml' run agent
```

看到 `Registered tunnel connection` 等连接成功日志后，访问 <https://agent.tanke.asia/>。浏览器使用公网 HTTPS 地址，本机回源仍为 HTTP；本例中二者之间的回源发生在同一台电脑上，不需要额外启动 README 中 Docker 的 Caddy HTTPS 服务。

首次配置完成后，以后演示只需重复本节的两个启动命令，域名保持不变。两个终端都要保持运行；按 `Ctrl+C` 可分别停止应用和隧道。电脑关机、休眠或断网后，公网体验会中断；需要长期在线时，应将应用和隧道一起交给服务管理器或部署到持续运行的服务器。

## 验收与常见问题

先在本机检查服务，再检查公网入口：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-RestMethod https://agent.tanke.asia/api/health
```

`/api/health` 用于健康检查，不需要访问密码；它返回正常只代表对应接口可达。还应在浏览器中确认密码保护生效、编辑器可用、PDF 可以打开、改写或 Agent 的内容能流式出现。完成配置后，可以用手机关闭 Wi-Fi，通过移动网络检查外部访问效果。

| 现象 | 检查方式 |
|---|---|
| 找不到 `cloudflared.exe` | 确认已进入仓库根目录，且本机存在 `tools\cloudflared.exe`；该工具不会随 Git 克隆 |
| 隧道或凭据不存在 | 检查 `config.yml` 的 UUID、凭据路径和当前 Windows 用户；自行部署需要自己创建隧道 |
| 未读取到配置或读错配置 | 使用上文带 `--config` 的命令指定完整路径 |
| DNS 已存在 | 检查 Cloudflare DNS 中的现有记录是否已经绑定正确隧道；已配置成功时跳过 DNS 步骤 |
| 公网页面返回 502 | 先检查本机 `/api/health` 是否正常，以及本地端口是否与 `service` 一致 |
| Cloudflare 提示隧道不可用 | 检查 `cloudflared` 进程和连接日志，确认源电脑没有休眠或断网 |
| 健康检查正常，但打开页面要求登录 | 设置 `ARIA_PASSWORD` 后的正常行为；在网页登录框中填写访问密码 |
| 页面能打开，但改写或 Agent 不流式输出 | 确认使用正式命名隧道；再查看应用日志和浏览器网络请求，检查是否还有会缓冲响应的代理 |

当前部署没有每人独立的账号和数据空间：获准进入同一实例的用户共享文献库与论文列表。体验时使用适合共享的数据，不要把入口密码当成文档隔离机制。Cloudflare Tunnel 提供公网入口，不保证中国大陆所有地区、运营商或校园网的访问质量，需要按实际使用网络测试。

## 为什么使用正式隧道

Ariadne 的 Agent、改写等功能依赖 SSE。Cloudflare 官方明确说明，随机 `trycloudflare.com` 地址的 Quick Tunnel 不支持 SSE，且地址会随重新创建变化，因此不作为本项目完整体验或海报二维码的推荐入口。本文使用绑定固定域名的正式命名隧道。

官方参考：[本地管理隧道的创建、DNS 和运行](https://developers.cloudflare.com/tunnel/features/locally-managed-tunnels/create-local-tunnel/)、[配置文件与 ingress 规则](https://developers.cloudflare.com/tunnel/features/locally-managed-tunnels/configuration-file/)、[Quick Tunnel 限制](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/#limitations)。
