#ifndef MAP_RECV_COMPONENT_H
#define MAP_RECV_COMPONENT_H

#include "component_factory.h"
#include "klog.h"
#include "node.h"

#include <iomanip>

#include "idmap_static.pb.h"
#include "ins_data.pb.h"
#include "file.h"

class map_recv_component : public Node {
 public:
  map_recv_component();
  ~map_recv_component();

 protected:
  virtual void init() override;

 private:
  void timerCallback();
  std::string recev_idmap_static_channel_;
  std::string recev_ins_channel_;

 public:
  void ReceiveStaticMap(const idmap::StaticIDMapInfo& data, double timestamp);
  void RecvInsData(const localization::InsData& ins_data, double timestamp);

  void configInit();
  void eventInit();
};

#endif  // MAP_RECV_COMPONENT_H
