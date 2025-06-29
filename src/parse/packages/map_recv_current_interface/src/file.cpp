#include "file.h"

#include <fcntl.h>
#include <glob.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <fstream>

INIT_LOG(map_static_info);

namespace common {

using std::istreambuf_iterator;
using std::string;
using std::vector;

// static bool raw_data_output = false;

// void SetOutputFlag(bool flag) { raw_data_output = flag; }

bool SetProtoToASCIIFile(const google::protobuf::Message &message,
                         int file_descriptor) {
  using google::protobuf::TextFormat;
  using google::protobuf::io::FileOutputStream;
  using google::protobuf::io::ZeroCopyOutputStream;
  if (file_descriptor < 0) {
    return false;
  }
  ZeroCopyOutputStream *output = new FileOutputStream(file_descriptor);
  bool success = TextFormat::Print(message, output);
  delete output;
  close(file_descriptor);
  return success;
}

bool SetProtoToASCIIFile(const google::protobuf::Message &message,
                         const std::string &file_name) {
  // if (!raw_data_output) {
  //   return false;
  // }
  int fd = open(file_name.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU);
  if (fd < 0) {
    return false;
  }
  return SetProtoToASCIIFile(message, fd);
}

bool GetProtoFromASCIIFile(const std::string &file_name,
                           google::protobuf::Message *message) {
  using google::protobuf::TextFormat;
  using google::protobuf::io::FileInputStream;
  using google::protobuf::io::ZeroCopyInputStream;
  int file_descriptor = open(file_name.c_str(), O_RDONLY);
  if (file_descriptor < 0) {
    // Failed to open;
    return false;
  }

  ZeroCopyInputStream *input = new FileInputStream(file_descriptor);
  bool success = TextFormat::Parse(input, message);
  if (!success) {
  }
  delete input;
  close(file_descriptor);
  return success;
}

bool GetProtoFromBinaryFile(const std::string &file_name,
                            google::protobuf::Message *message) {
  std::fstream input(file_name, std::ios::in | std::ios::binary);
  if (!input.good()) {
    return false;
  }
  if (!message->ParseFromIstream(&input)) {
    return false;
  }
  return true;
}

bool GetProtoFromJsonFile(const std::string &file_name,
                          google::protobuf::Message *message) {
    std::ifstream in(file_name);
    if (in.fail()) {
        KLOG_ERROR("open file failed");
        return false;
    }
    std::string content((istreambuf_iterator<char>(in)),
                        istreambuf_iterator<char>());
    google::protobuf::util::JsonParseOptions parseOpt;
    // parseOpt.ignore_unknown_fields = true;
    google::protobuf::StringPiece sp(content);
    google::protobuf::util::Status sts =
        google::protobuf::util::JsonStringToMessage(sp, message, parseOpt);
    if (!sts.ok()) {
        KLOG_ERROR("JsonStringToMessage error:",sts.ToString());
        return false;
    }
    return true;
}

bool GetProtoFromFile(const std::string &file_name,
                      google::protobuf::Message *message) {
  if (file_name.substr(file_name.find_last_of(".") + 1) == "json") {
    return GetProtoFromJsonFile(file_name, message);
  } else if (file_name.substr(file_name.find_last_of(".") + 1) == "bin") {
    return GetProtoFromBinaryFile(file_name, message);
  } else {
    return GetProtoFromASCIIFile(file_name, message);
  }
}

}  // namespace common
