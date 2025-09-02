#!/bin/bash

# Target value
NEW_LIMIT=524288

echo "🔧 Increasing inotify watch limit to $NEW_LIMIT..."

# 1. Apply temporary change
sudo sysctl fs.inotify.max_user_watches=$NEW_LIMIT

# 2. Make it permanent in /etc/sysctl.conf
SYSCTL_CONF="/etc/sysctl.conf"
if grep -q "fs.inotify.max_user_watches" $SYSCTL_CONF; then
    echo "📝 Updating existing entry in $SYSCTL_CONF"
    sudo sed -i "s/^fs.inotify.max_user_watches=.*/fs.inotify.max_user_watches=$NEW_LIMIT/" $SYSCTL_CONF
else
    echo "📌 Appending setting to $SYSCTL_CONF"
    echo "fs.inotify.max_user_watches=$NEW_LIMIT" | sudo tee -a $SYSCTL_CONF
fi

# 3. Reload sysctl settings
echo "🔄 Reloading sysctl settings..."
sudo sysctl -p

# 4. Optional: disable Streamlit file watcher
read -p "⚙️  Do you want to disable Streamlit file watcher as well? (y/n): " disable_streamlit

if [[ "$disable_streamlit" == "y" ]]; then
    STREAMLIT_CONFIG_DIR="$HOME/.streamlit"
    STREAMLIT_CONFIG_FILE="$STREAMLIT_CONFIG_DIR/config.toml"

    mkdir -p "$STREAMLIT_CONFIG_DIR"
    if grep -q "fileWatcherType" "$STREAMLIT_CONFIG_FILE"; then
        sed -i 's/^fileWatcherType *= *.*/fileWatcherType = "none"/' "$STREAMLIT_CONFIG_FILE"
    else
        echo -e "\n[server]\nfileWatcherType = \"none\"" >> "$STREAMLIT_CONFIG_FILE"
    fi
    echo "✅ Streamlit fileWatcherType set to 'none' in $STREAMLIT_CONFIG_FILE"
fi

echo "✅ All done! You may need to restart apps or terminals."
