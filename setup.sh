curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env 
curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.tar.gz
rm -rf /opt/nvim-linux-x86_64
tar -C /opt -xzf nvim-linux-x86_64.tar.gz
export PATH="$PATH:/opt/nvim-linux-x86_64/bin"
uv venv
uv pip install -r requirements.txt
uv run python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('IlyaGusev/gemma-2-9b-it-abliterated'); AutoModelForCausalLM.from_pretrained('IlyaGusev/gemma-2-9b-it-abliterated', torch_dtype='auto', device_map='auto')"
