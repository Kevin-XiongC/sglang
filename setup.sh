git remote add xc git@github.com:Kevin-XiongC/sglang.git
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple
git fetch xc
git checkout xc
pip install -e "python[all]"
mkdir ~/kimi
ln -s /model/data2/Kimi-K2-Instruct-0905/* ~/kimi/