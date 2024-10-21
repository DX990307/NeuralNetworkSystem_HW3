#pragma once

#include <string>

using std::string;
typedef uint32_t vid_t;

class graph_t {
public:
    int* offset_csr;
    int* nebrs_csr;
    int* offset_csc;
    int* nebrs_csc;
    vid_t vcount;
    vid_t ecount;

    void init(vid_t a_vcount, vid_t a_dstsize, void* a_offset, void* a_nebrs, 
              void* a_offset1, void* a_nebrs1, int64_t a_flag, vid_t edge_count) {
        offset_csr = static_cast<int*>(a_offset);
        nebrs_csr = static_cast<int*>(a_nebrs);
        offset_csc = static_cast<int*>(a_offset1);
        nebrs_csc = static_cast<int*>(a_nebrs1);
        vcount = a_vcount;
        ecount = edge_count;
    };
    void save_graph(const string& full_path) {};
    void load_graph(const string& full_path) {};
    void load_graph_noeid(const string& full_path) {};
    int get_vcount() { return vcount; };
    int get_ecount() { return ecount; };
};
