function print_json(name, pid, tid, ts, dur, args_str, start_ts) {
  if (ts != 0 && dur != 0 && ts > start_ts) {
    printf("{ \"name\": \"%s\", \"ph\": \"X\", \"pid\": %d, \"tid\": %d, \"ts\": %d, \"dur\": %d%s },\n",
        name, pid, tid, ts, dur, args_str);
  }
}

function print_meta(name, pid, tid) {
  if (tid == "") {
    printf("{ \"name\": \"process_name\", \"ph\": \"M\", \"args\": { \"name\": \"%s\" }, \"pid\": %d },\n", name, pid);
  }
  else {
    printf("{ \"name\": \"thread_name\", \"ph\": \"M\", \"args\": { \"name\": \"%s\" }, \"pid\": %d, \"tid\": %d },\n", name, pid, tid)
  }
}

function create_args_str(args, nargs, data, extra) {
  args_str = "";
  if (nargs != 0) {
    for (i = 0; i < nargs; i++) {
      args_str = args_str"\""args[i]"\": \""data[args[i]]"\"";
      if (i != nargs - 1)
        args_str = args_str", ";
    }
  }
  if (extra != "") {
    if (args_str != "")
      args_str = args_str", "extra;
    else
      args_str = extra;
  }
  if (args_str != "")
    args_str = ", \"args\": { "args_str" }";

  return args_str;
}

function create_arg_item(name, value) {
  return "\""name"\": \""value"\"";
}

function read_data(data) {
  delete data;
  for (i = 1; i <= NF; i++) {
    split($i, tmp, " ");
    data[tmp[1]] = tmp[2];
  }
}

function hex2dec(hex) {
  hex_chars="123456789abcdef";
  split(hex, chars, "")
  mul = 1;
  result = 0;
  for (i = length(hex); i >= 1; i--) {
    val = index(hex_chars, chars[i]);
    result += val * mul;
    mul *= 16;
  }
  return result;
}
