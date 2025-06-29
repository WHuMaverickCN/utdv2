# from proto.proto_py import ins_data_pb2
#
#
# def parser_ins(topic_size, topic_content):
#     data=topic_content[:topic_size]
#     ins_struct  = ins_data_pb2.InsData()
#     ins_res = ins_struct.ParseFromString(data)
#     print(ins_res)
#     if ins_res:
#         return True,ins_res
#
#     print(id.temperature)
