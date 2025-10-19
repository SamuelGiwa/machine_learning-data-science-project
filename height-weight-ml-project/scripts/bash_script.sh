#!/bin/bash

# this scripts connects with an API created using FASTAPI

URL="http://127.0.0.1:8000/predict_height" #API URL

extract_height_prediction() {
    echo "$1" | grep -o '"height_prediction":[^,}]*' | awk -F: '{print $2}' | tr -d ' '
}


INPUT_FILE="weights.txt"   # a text file containning weights


OUTPUT_FILE="height_predictions.csv"


echo "Weight,PredictedHeight" > "$OUTPUT_FILE"


while IFS= read -r weight; do   # -r prevents backlash escape; output stored in weight 
    
    weight=$(echo "$weight" | tr -d '\r')
   

    
    response=$(curl -s -X POST "$URL" \
                  -H "Content-Type: application/json" \
                  -d "{\"Weight\": $weight}")
    height=$(extract_height_prediction "$response")

    
    printf "%s,%s\n" "$weight" "$height" >> "$OUTPUT_FILE"


done < "$INPUT_FILE"

echo "All results saved to $OUTPUT_FILE"