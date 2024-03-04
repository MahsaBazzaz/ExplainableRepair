#!/bin/bash
#
# to run:
# chmod +x job.sh
# ./job.sh
#
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <parameter1> <parameter2>"
  exit 1
fi
game=$1

root_folder="/out/$game/$index"
num_runs=100
echo "<><><><><> Parameter 1: game:  $game"
echo "<><><><><> root_folder $root_folder"

#
# cave setting
#
if [ "$game" == "cave" ]; then
    size="15 12"
    reachmove="maze"
    reachstartgoal="tl-br 5"
    pattern="nbr-plus"
    solvers="pysat-rc2 scipy pysat-rc2-boolonly"
    tilefile="$root_folder/$game.tile"
    schemefile="$root_folder/$game.scheme"
    tiemout=300
    command="python3 sturgeon/input2tile.py --outfile $tilefile --textfile ./sturgeon/levels/kenney/cave.lvl --imagefile ./sturgeon/levels/kenney/cave.png --tilesize 16 --quiet"
    $command
    command="python3 sturgeon/tile2scheme.py --outfile $schemefile --tilefile $tilefile --pattern $pattern --count-divs 1 1 --quiet"
    $command
fi
#
# mario setting
#
if [ "$game" ==  "mario" ]; then
    size="14 18"
    reachmove="platform"
    reachstartgoal="l-r 4"
    pattern="ring"
    solvers="scipy pysat-rc2-boolonly pysat-rc2"
    tilefile="$root_folder/$game.tile"
    schemefile="$root_folder/$game.scheme"
    tiemout=5000
    command="python3 sturgeon/input2tile.py --outfile $tilefile --textfile ./sturgeon/levels/vglc/mario-1-1-generic.lvl --quiet"
    $command
    command="python3 sturgeon/tile2scheme.py --outfile $schemefile --tilefile $tilefile --pattern $pattern  --count-divs 1 1 --quiet"
    $command
fi
#
# supercat setting
#
if [ "$game" ==  "supercat" ]; then
    size="20 20"
    reachmove="supercat"
    reachstartgoal="b-t 8"
    pattern="diamond"
    solvers="pysat-rc2 scipy pysat-rc2-boolonly"
    tilefile="$root_folder/$game.tile"
    schemefile="$root_folder/$game.scheme"
    tiemout=4000
    command="python3 sturgeon/input2tile.py --outfile $tilefile --textfile ./sturgeon/levels/manual/supercat-1-7.lvl --tilesize 12 --quiet"
    $command
    command="python3 sturgeon/tile2scheme.py --outfile $schemefile --tilefile $tilefile --pattern $pattern  --count-divs 5 5 --quiet"
    $command
fi
# Check if the parent directory exists
if [ ! -d "$root_folder" ]; then
    echo "Parent directory does not exist."
    exit 1
fi

directory_count=$(find "$root_folder" -mindepth 1 -maxdepth 1 -type d | wc -l)
current_number=$(($directory_count+1))

RANDOM=$$
random_value=$(($RANDOM))
full_path="$root_folder/$current_number"
textfile="$full_path/$current_number.lvl"
start_end="$full_path/start_end_$current_number.txt"

deep_shap_weight="$full_path/deep_shap_$current_number.json"
deep_shap_weight_orig="$full_path/orig_deep_shap_$current_number.json"
deep_shap_image="$full_path/shap_$current_number.png"
deep_shap_repaired="$full_path/shap_$current_number.repaired"
deep_shap_path="$full_path/shap_path"

ig_weight="$full_path/ig_$current_number.json"
ig_weight_orig="$full_path/orig_ig_$current_number.json"
ig_image="$full_path/ig_$current_number.png"
ig_repaired="$full_path/ig_$current_number.repaired"
ig_path="$full_path/ig_path"

uni_weight="$full_path/uni_$current_number.json"
uni_repaired="$full_path/uni_$current_number.repaired"
uni_path="$full_path/uni_path"

if [ ! -d "$full_path" ]; then
    mkdir "$full_path"
fi

#
# generate an unplayble level
#
command="python3 sturgeon/scheme2output.py --outfile $full_path/$current_number --schemefile $schemefile --size $size --reach-start-goal $reachstartgoal --reach-move $reachmove --reach-unreachable --random $random_value --quiet --pattern-hard --count-soft"
$command
#
# start end hard constraint
#
command="python3 utils/start_end.py --input $textfile --out $start_end"
$command
#
# ig weight
#
command="python3 explainers/ig.py --game $game --outfile $ig_weight_orig $ig_weight --level $textfile"
$command | tail -n 1 > $full_path/ig_weight_time.txt
#
# deep shapley weight
#
command="python3 explainers/deep_shap.py --game $game --outfile $deep_shap_weight_orig $deep_shap_weight --level $textfile"
$command | tail -n 1 > $full_path/deep_shap_weight_time.txt
#
# uniform weight
#
command="python3 utils/uniform_weight.py --outfile $uni_weight --game $game"
$command
#
# ig repair
#
command="python3 sturgeon/scheme2output.py --outfile $ig_outputfile --schemefile $schemefile --size $size --reach-move $reachmove --reach-start-goal $reachstartgoal --custom text-level-weighted $textfile $ig_weight --custom text-level $start_end hard --solver $solvers --pattern-hard"
timeout $tiemout $command | tee >(tail -n 1 > $full_path/ig_time.txt) > $full_path/ig_log.txt
#
# deep shap repair
#
command="python3 sturgeon/scheme2output.py --outfile $outputfile_shapley_1 --schemefile $schemefile --size $size --reach-move $reachmove --reach-start-goal $reachstartgoal --custom text-level-weighted $textfile $deep_shap_weight --custom text-level $start_end hard --solver $solvers --pattern-hard"
timeout $tiemout $command | tee >(tail -n 1 > $full_path/deep_shap_time.txt) > $full_path/deep_shap_log.txt
#
# uniform repair
#
wait $pid3
command="python3 sturgeon/scheme2output.py --outfile $outputfile_uniform --schemefile $schemefile --size $size --reach-move $reachmove --reach-start-goal $reachstartgoal --custom text-level-weighted $textfile $uni_weight --custom text-level $start_end hard --solver $solvers --pattern-hard"
timeout $tiemout $command | tee >(tail -n 1 > $full_path/uniform_time.txt) > $full_path/uni_log.txt





