cd packages/map_recv_current_interface

# 从 dat_files.json 中读取多个路径
dat_files=$(jq -r '.[]' ../../dat_files.json)

# 遍历每个路径并逐个解析
for CAPILOT_INPUT_DIR in $dat_files; do
    echo "正在解析文件: $CAPILOT_INPUT_DIR"
    
    # 原有的解析逻辑
    # 检查当前目录是否存在data.csv文件，如果存在则先删除
    if [ -f "data.csv" ]; then
        chmod +w "data.csv" 2>/dev/null
        rm -f "data.csv" --force
        echo "已删除旧的 data.csv 文件"
    fi
    sudo capilot --task_path=./task.yaml --f="$CAPILOT_INPUT_DIR" 2>&1 | while read -r line; do
        echo "$line"

        # 捕获到 "replay over" 时终止 capilot 进程
        if echo "$line" | grep -q "replay over"; then
            echo "检测到控制台 'replay over',组合导航模块解析完毕,退出Capilot工具..."
            sudo pkill -f "capilot"
            break
        fi

        # 只在特定条件下写入日志,避免写入过多的无关日志
        if echo "$line" | grep -q "ERROR"; then
            echo "[错误日志] -- $line"
        fi
    done

    # 根据CAPILOT_INPUT_DIR的文件名，生成目标文件名
    base_filename=$(basename "$CAPILOT_INPUT_DIR" .dat)
    target_filename="${base_filename}.csv"

    # 将当前文件夹下的data.csv另存为目标文件名
    pwd
    if [ -f "data.csv" ]; then
        mv "data.csv" "$target_filename"
        echo "已将data.csv另存为 $target_filename"
    else
        echo "未找到data.csv文件,无法另存为 $target_filename"
    fi
done

# 从 config.yaml 中读取 data_pool_dir
data_pool_dir=$(grep "dataset_path:" ../../config.yaml | awk -F': ' '{print $2}')
data_pool_dir="${data_pool_dir}/location"
echo "目标目录: $data_pool_dir"

# 检查目标目录是否存在，不存在则创建
if [ ! -d "$data_pool_dir" ]; then
    mkdir -p "$data_pool_dir"
    echo "创建目标目录: $data_pool_dir"
fi

# 查找符合命名规则的文件并移动到目标目录
for file in CD701_*.csv; do
    if [ -f "$file" ]; then
        mv "$file" "$data_pool_dir"
        echo "已将文件 $file 移动到 $data_pool_dir"
    else
        echo "未找到符合命名规则的文件"
    fi
done
