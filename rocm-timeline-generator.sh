#!/bin/bash
#
# rocm-timeline-generator.sh
#
# Created by Sarunya Pumma
#
# Format profiles from TensorFlow tracer, HCC profiler and HIP
# profiler, and custom timers in TensorFlow in chrome://tracing/
# JSON format
#
# This script requires fn-rocm-timeline-generator.awk which contains
# awk functions for printing data in JSON format
#

get_arg () {
  if [ -z "$1" ]
  then
    # default value
    echo $2
  else
    echo $1
  fi
}

usage="$0 <log file> <output filename> <optional: TF timeline prefix i.e., tf_cifar10_timeline> <optional: start time> <optional: end time>"

: ${1?$usage}
: ${2?$usage}

log_file=$1
output=$2

tf_tl_prefix=`get_arg $3 "_none"`
start_time=`get_arg $4 -1`
end_time=`get_arg $5 -1`

tmp=/tmp
awk_fn=fn-rocm-timeline-generator.awk
AWK=gawk

echo "log file ${log_file}"

declare -A timeline

files="tf hcc hip rccl custom_tf"
for file in ${files}; do
  timeline[$file]="${tmp}/timeline_${file}.json"
  rm -f ${timeline[$file]}
done;

count_timeline=`ls ${tf_tl_prefix}_*.json 2> /dev/null | wc -l`

num_meta=0

echo "formating TF timeline"
if [[ "$count_timeline" -ge "1" ]]; then 
  # compute number of lines that contain metadata events
  num_meta_lines=`cat ${tf_tl_prefix}_0.json | ${AWK} '
    NR == 1 {
      lines = 2;
    }
    NR > 2 {
      lines++;
      open_brackets = gsub(/{/, "{");
      close_brackets = gsub(/}/, "}");
      count += open_brackets - close_brackets;
      if ($0 ~ /ph/ && $NF !~ /\"M\"/) done = 1;
      if (count == 0 && done == 0) last_meta_line = lines;
    }
    END {
      print last_meta_line;
    }'`

  # create metadata dictionary (mapping pid to name)
  head -${num_meta_lines} ${tf_tl_prefix}_0.json | \
    ${AWK} 'NR > 2 {
      open_brackets = gsub(/{/, "{");
      close_brackets = gsub(/}/, "}");
      count += open_brackets - close_brackets;
      if (count == 0)
        printf("%d\t%s\n", pid, name);
      if (count == 2 && $0 ~ /name/) {
        split($0, tmp, ": ");
        name = tmp[2]; 
      }
      else if (count == 1 && $0 ~ /pid/) {
        split($NF, tmp, ",");
        pid = tmp[1];
      }
    }' > ${tmp}/dict;

  num_meta=`cat ${tmp}/dict | wc -l`

  # change metadata of all JSON files to be the same
  for (( i = 0 ; i < $count_timeline; i++ )); do
    file=${tf_tl_prefix}_${i}.json;
    head -${num_meta_lines} ${tmp}/dict $file | \
      ${AWK} -v num_meta=${num_meta} 'NR >= 2 && NR <= (num_meta + 1) {
        split($0, tmp, "\t");
        dict[tmp[2]] = tmp[1];
      }
      NR > (num_meta + 5) {
        open_brackets = gsub(/{/, "{");
        close_brackets = gsub(/}/, "}");
        count += open_brackets - close_brackets;
        if (count == 0)
          printf("from %d\tto %d\n", pid, dict[name])
        if (count == 2 && $0 ~ /name/) {
          split($0, tmp, ": ");
          name = tmp[2];
        }
        else if (count == 1 && $0 ~ /pid/) {
          split($NF, tmp, ",");
          pid = tmp[1];
        }
      }' > ${tmp}/changes;
    sed -i 's/"pid": /NEED_CHANGE/g' $file;
    while read change; do
      from_id=`echo $change | ${AWK} '{print $2}'`;
      to_id=`echo $change | ${AWK} '{print $NF}'`;
      sed -i "s/NEED_CHANGE${from_id}\([,]\?\)$/\"pid\": ${to_id}\1/g" $file;
    done < ${tmp}/changes;
  done

  # concat timeline files
  for (( i = 0; i < $count_timeline; i++ )); do
    file=${tf_tl_prefix}_${i}.json;
    lines=`cat $file | wc -l`;
    cat $file | \
      tail -`echo "$lines - 1" | bc` | \
      head -`echo "$lines - 3" | bc`;
    echo ",";
  done > ${timeline["tf"]}
fi

# compute a difference between the machine time and GPU time
diff=`grep -a "hcc-ts-ref, prof_name gpu_host_ts" $log_file | \
  ${AWK} -e '{
    read_data(data);
    host_time = data["unix_ts"];
    gpu_time = data["gpu_ts"];
    printf("%d\n", host_time - (gpu_time / 1e3))
  }' -f $awk_fn -F ', ' | head -1`

echo "formating HIP timeline"
num_meta=`echo "${num_meta} + 1" | bc`

tmp_toskip=${tmp}/toskip
tmp_hip_api=${tmp}/hip-api

grep -a "hip-api" $log_file | \
  sed 's/.*\(hip-api\)/\1/g' | \
  ${AWK} '{print $2}' | \
  sort | uniq -c | sort -nr | \
  ${AWK} '{if ($2 ~ /tid/ && $2 !~ /HIP/ && $1 != 2) print "hip-api toskip "$2}' \
    > $tmp_toskip

cat $tmp_toskip $log_file | \
  grep -a "hip-api" | \
  sed 's/.*\(hip-api\)/\1/g' | \
  ${AWK} '{
    if ($2 ~ /toskip/)
      toskip[$3] = 1;
    else {
      tid = $2;
      if ((tid ~ /tid/ && toskip[tid] != 1) || $0 ~ /HIP initialized/)
        print $0
      }
    }' > $tmp_hip_api

# format timeline from hip-api
for tid in `/bin/grep -a "HIP initialized" $tmp_hip_api | \
  ${AWK} -F ':' '{printf("%d ", $2)}'`; do
  grep -a "tid:${tid}[.]" $tmp_hip_api | \
    ${AWK} -e 'NR == 1 { lines = 0; }
  {
    lines++;
    name = $3;
    if (lines % 2 == 1) {
      split($NF, tmp, "@");
      ts = (tmp[2] / 1e3) + diff;
      parm = "";
      if (NF != 4)
        parm = $4;
      for ( i = 5; i < NF; i++ )
        parm = parm" "$i;
      gsub(/, /, "\\n", parm);
    }
    else {
      split($(NF-1), tmp, "+");
      dur = tmp[2] / 1e3;
      status = gensub("[(](.*)[)]>>", "\\1", "g", $(NF - 2));
      create_args_str("", 0, "", "\"status\": \""status"\", \"parm\": \""parm"\"");
      print_json(name, pid, tid, ts, dur, args_str, start_ts, end_ts);
    }
  }' -f $awk_fn -v diff=$diff -v pid=$num_meta -v tid=$tid \
    -v start_ts=${start_time} -v end_ts=${end_time} \
    >> ${timeline["hip"]}
  grep -a "tid:${tid}:" $tmp_hip_api | \
    grep -a "HIP initialized" | \
    ${AWK} '{ print $NF }' | \
    sed 's/0x\(.*\))/\1/g' | \
    ${AWK} -e '{ hex = hex2dec($0); print_meta(hex, pid, tid) }' \
      -f $awk_fn -v pid=$num_meta -v tid=$tid \
      >> ${timeline["hip"]}
done;

# print canSeeMemory function in details, while printing only the total time of
# other profiles
#grep -a "hip-profile" $log_file | \
#  awk -e '{
#    read_data(data);
#    tid = data["id"];
#    if (data["prof_name"] ~ /canSeeMemory/) {
#      args[1]="dstCtx"; args[2]="srcCtx"; nargs=2
#      args_str = create_args_str(args, nargs, data);
#      print_json("canSeeMemory, check dst", pid, tid,
#        data["ts_check_dst"], data["check_dst_time"], args_str,
#        start_ts, end_ts);
#      print_json("canSeeMemory, lock dst", pid, tid,
#        data["ts_lock_dst"], data["lock_dst_time"], args_str,
#        start_ts, end_ts);
#      print_json("canSeeMemory, check src", pid, tid,
#        data["ts_check_src"], data["check_src_time"], args_str,
#        start_ts, end_ts);
#      print_json("canSeeMemory, lock src", pid, tid,
#        data["ts_lock_src"], data["lock_src_time"], args_str,
#        start_ts, end_ts);
#    }
#    else {
#      args[1] = "args"; nargs = 1;
#      args_str = create_args_str(args, nargs, data);
#      print_json(data["prof_name"], pid, tid, data["ts"],
#        data["time"], args_str, start_ts, end_ts);
#    }
#  }
#  END {
#    print_meta("hip-profile", pid);
#  }' -f $awk_fn -F ', ' -v pid=${num_meta} -v start_ts=${start_time} \
#    -v end_ts=${end_time} \
#    >> ${timeline["hip"]}

echo "formating HCC timeline"
num_meta=`echo "${num_meta} + 1" | bc`

# format HCC profiles
grep -a "profile:" $log_file | \
  sed 's/;//g' | \
  ${AWK} -e  'NR == 1 { max_dev = pid; }
  {
    split($8, id, ".");
    split(id[1], dev, "#");
    name = $3;
    args = create_arg_item("device.queue.cmd", $8);
    if ($2 ~ /copy/) name = $3"_"$9;
    else if ($2 ~ /barrier/ && $0 ~ /deps/) {
      split($0, deps, "deps=");
      name = $3;
      args = args", "create_arg_item("deps", deps[2]);
    }
    args_str = create_args_str("", 0, "", args);
    cur_dev = dev[2] + pid;
    if (cur_dev > max_dev) max_dev = cur_dev;
    print_json(name, cur_dev, id[2], ($6/1e3) + diff, $4, args_str,
      start_ts, end_ts);
  }
  END {
    for (i = pid; i <= max_dev; i++)
      print_meta("hcc-profile, device "(i - pid), i);
    print (max_dev - pid);
  }' -v diff=${diff} -f $awk_fn -v pid=${num_meta} \
     -v start_ts=${start_time} -v end_ts=${end_time} \
    > ${timeline["hcc"]}

# get pid_offset from the last line and remove the last line from the file
pid_offset=`echo $(tail -1 ${timeline["hcc"]}) | \
  ${AWK} '{if ($0 == "") print 0; else print $0}'`
sed -i '$d' ${timeline["hcc"]}

num_meta=`echo "${num_meta} + ${pid_offset} + 1" | bc`

grep -a "hcc-profile" $log_file | \
  ${AWK} -e '{
    read_data(data);
    name = data["prof_name"];
    if (get_tid[name] == "") {
      get_tid[name] = tid;
      print_meta(name, pid, tid);
      tid++;
    }
    this_tid = get_tid[name];
    print_json(name, pid, this_tid, data["ts"], data["time"], "",
      start_ts, end_ts);
  }
  END {
    print_meta("hcc-profile, runtime", pid);
  }
  ' -f $awk_fn -F ', ' -v pid=${num_meta} -v start_ts=${start_time} \
    -v end_ts=${end_time} >> ${timeline["hcc"]}

echo "formating RCCL timeline"
num_meta=`echo "${num_meta} + 1" | bc`

grep -a "rccl-api" $log_file | \
  awk -e '{
    read_data2(data);
    name = data["func"];
    this_tid = data["device"];
    args_str = create_args_str(args, nargs, data);
    print_json(name, pid, this_tid, data["ts"], data["dur"], args_str,
      start_ts, end_ts);
  }
  END {
    print_meta("rccl-profile, runtime", pid);
  }
  ' -f $awk_fn -F ' ' -v pid=${num_meta} -v start_ts=${start_time} \
    -v end_ts=${end_time} >> ${timeline["rccl"]}

echo "formating custom TF timers"

num_meta=`echo "${num_meta} + 1" | bc`

# note: data augmentation profile is specific for cifar_multi_gpu_train.py
grep -a "tf-profile" $log_file | \
  ${AWK} -e 'NR == 1 { tid_count = 0 }
  {
    data["args"] = "";
    read_data(data);
    ts = data["ts"];
    time = data["time"];
    tid = 0;
    if (data["prof_name"] == "process") {
      args[1]="name"; args[2]="op"; args[3]="device";
      args[4]="step"; args[5]="type"; nargs=5;
      args_str = create_args_str(args, nargs, data);
      if (data["type"] == "DATA") {
        name = "Data augmentation";
        if (get_tid[name] == "") {
          get_tid[name] = tid_count;
          print_meta("Data augmentation", pid, tid_count);
          tid_count++;
        }
        tid = get_tid[name];
      }
      else {
        if (data["op"] == "") {
          name = "Process";
        }
        else {
          name = data["op"];
        }
        tid = data["id"];
      }
    }
    else {
      name = data["prof_name"];
      nargs = split(data["args"], args, ";");
      args_str = create_args_str(args, nargs, data);
      if (data["id"] == "") {
        if (get_tid[name] == "") {
          get_tid[name] = tid_count;
          print_meta(name, pid, tid_count);
          tid_count++;
        }
        tid = get_tid[name];
      }
      else
        tid = data["id"];
    }
    print_json(name, pid, tid, ts, time, args_str, start_ts, end_ts);
  }
  END {
    print_meta("tf-profile", pid);
  }' -f $awk_fn -F ', ' -v pid=${num_meta} -v start_ts=${start_time} \
    -v end_ts=${end_time} \
    > ${timeline["custom_tf"]}

# merge timeline files
final_timeline=${tmp}/timeline.json
rm -f $final_timeline
echo -e "{\n\t\"traceEvents\": [" > $final_timeline
for file in $files; do
  cat ${timeline[$file]} >> $final_timeline
done;

# get rid of comma in the last line
last_line=`tail -1 $final_timeline | sed 's/,$//g'`
sed -i '$d' $final_timeline
echo $last_line >> $final_timeline 
echo -e "\t]\n}" >> $final_timeline

mv $final_timeline $output

echo "done, output $output"
