
新机安装清单（拷贝之后做这4步）

# 0. 前提：serv00 域名已添加（如 xxx.serv00.com），git clone 仓库到域名目录
cd ~/domains/<你的域名>
git clone https://github.com/jwfst5088/gd-vps.git tmp && mv tmp/* tmp/.* . 2>/dev/null; rm -rf tmp

# 1. 编译（首次拉依赖，几分钟）
cd clawguandan && cargo build --release

# 2. 启动守护进程
/usr/sbin/daemon -f -p /tmp/clawguandan.pid ./target/release/clawguandan server serve --ip 0.0.0.0 --port 2230

# 3. crontab -e 加入保活（每5分钟）
*/5 * * * * if [ -f /tmp/clawguandan_stop ]; then exit 0; fi; curl -sf --connect-timeout 5 http://localhost:2230/ping > /dev/null 2>&1 || /usr/sbin/daemon -f -p /tmp/clawguandan.pid /usr/home/$USER/domains/<你的域名>/clawguandan/target/release/clawguandan server serve --ip 0.0.0.0 --port 2230

# 4.（可选）加入每3天学习闭环
9 3 */3 * * /bin/sh /home/<用户>/domains/<你的域名>/clawguandan/auto_learn.sh
