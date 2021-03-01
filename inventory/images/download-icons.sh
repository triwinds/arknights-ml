curl -L https://github.com/penguin-statistics/frontend-v2/archive/dev.zip  --output ./frontend-v2-dev.zip
"C:/Program Files/7-Zip/7z" e ./frontend-v2-dev.zip -o./icon "*/ios/App/PenguinWidget/ItemIcons.xcassets/*/*.png" -r -y
rm ./frontend-v2-dev.zip