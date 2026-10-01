// Exact, dependency-free small-graph enumeration for the radius-two atlas.
// Build: c++ -O3 -std=c++17 atlas.cpp -o /tmp/radius-two-atlas
// Run:   /tmp/radius-two-atlas MAX_VERTICES MAX_DEGREE OUTPUT_DIRECTORY
// Rooted keys are (vertex count, adjacency mask), with vertex zero the root.
// The mask enumerates pairs (0,1),(0,2),...,(n-2,n-1), least bit first.
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;
using Mask = uint64_t;
using Key = pair<int, Mask>;
using Partition = vector<vector<int>>;
struct Graph {
    int n = 0;
    array<unsigned, 11> a{};
    void edge(int u, int v) { a[u] |= 1u << v; a[v] |= 1u << u; }
    int degree(int u) const { return __builtin_popcount(a[u]); }
    int maxdegree() const {
        int d = 0; for (int u=0;u<n;++u) d=max(d,degree(u)); return d;
    }
};
Graph decode(Key k) {
    Graph g; g.n=k.first; int bit=0;
    for(int u=0;u<g.n;++u) for(int v=u+1;v<g.n;++v,++bit)
        if(k.second & (Mask(1)<<bit)) g.edge(u,v);
    return g;
}
Mask encode(const Graph& g, const vector<int>& order) {
    Mask m=0; int bit=0;
    for(int u=0;u<g.n;++u) for(int v=u+1;v<g.n;++v,++bit)
        if(g.a[order[u]] & (1u<<order[v])) m |= Mask(1)<<bit;
    return m;
}
Partition refine(const Graph& g, Partition p) {
    while(true) {
        vector<unsigned> masks;
        for(auto& cell:p) { unsigned m=0; for(int v:cell)m|=1u<<v; masks.push_back(m); }
        Partition next;
        for(auto& cell:p) {
            map<vector<int>,vector<int>> groups;
            for(int v:cell) {
                vector<int> signature;
                for(unsigned m:masks)signature.push_back(__builtin_popcount(g.a[v]&m));
                groups[signature].push_back(v);
            }
            for(auto& group:groups)next.push_back(group.second);
        }
        if(next.size()==p.size())return next;
        p=move(next);
    }
}
Mask canon_search(const Graph& g, Partition p) {
    p=refine(g,move(p));
    int split=-1;
    for(int i=0;i<(int)p.size();++i)if(p[i].size()>1){split=i;break;}
    if(split<0){vector<int> order;for(auto& c:p)order.push_back(c[0]);return encode(g,order);}
    Mask best=~Mask(0); vector<int> tried;
    for(int v:p[split]) {
        bool twin=false;
        for(int u:tried)if((g.a[v]&~(1u<<u))==(g.a[u]&~(1u<<v))){twin=true;break;}
        if(twin)continue; // The transposition fixes every other vertex.
        tried.push_back(v);
        Partition child=p; vector<int> rest;
        for(int u:p[split])if(u!=v)rest.push_back(u);
        child[split]={v}; child.insert(child.begin()+split+1,rest);
        best=min(best,canon_search(g,move(child)));
    }
    return best;
}
Key canonical(const Graph& g, const vector<int>& fixed={}) {
    Partition p; unsigned used=0;
    for(int v:fixed){p.push_back({v});used|=1u<<v;}
    vector<int> rest;for(int v=0;v<g.n;++v)if(!(used&(1u<<v)))rest.push_back(v);
    if(!rest.empty())p.push_back(rest);
    return {g.n,canon_search(g,move(p))};
}
vector<int> distances(const Graph& g,int root) {
    vector<int> d(g.n,-1),q{root};d[root]=0;
    for(size_t i=0;i<q.size();++i)for(int v=0;v<g.n;++v)
        if((g.a[q[i]]&(1u<<v))&&d[v]<0){d[v]=d[q[i]]+1;q.push_back(v);}
    return d;
}
Graph induced(const Graph& g,const vector<int>& order) {
    Graph h;h.n=order.size();
    for(int u=0;u<h.n;++u)for(int v=u+1;v<h.n;++v)
        if(g.a[order[u]]&(1u<<order[v]))h.edge(u,v);
    return h;
}
Key ball(const Graph& g,int root,int radius) {
    auto d=distances(g,root);vector<int> order{root};
    for(int v=0;v<g.n;++v)if(v!=root&&d[v]>=0&&d[v]<=radius)order.push_back(v);
    return canonical(induced(g,order),{0});
}
Graph local_product(const Graph& a,const Graph& b) {
    auto da=distances(a,0),db=distances(b,0);
    vector<pair<int,int>> vertices;
    for(int u=0;u<a.n;++u)for(int v=0;v<b.n;++v)
        if(da[u]+db[v]<=2)vertices.push_back({u,v});
    Graph c;c.n=vertices.size();
    if(c.n>11)throw runtime_error("product exceeds graph storage");
    for(int i=0;i<c.n;++i)for(int j=i+1;j<c.n;++j){
        auto [u,v]=vertices[i];auto [x,y]=vertices[j];
        if((u==x&&(b.a[v]&(1u<<y)))||(v==y&&(a.a[u]&(1u<<x))))c.edge(i,j);
    }
    return c;
}
int main(int argc,char** argv) {
    try {
        if(argc==2&&string(argv[1])=="canonical") {
            int n,k;
            while(cin>>n>>k){
                if(n<1||n>11||k<0||k>n)throw runtime_error("invalid canonical request");
                Graph g;g.n=n;for(int u=0;u<n;++u)cin>>g.a[u];
                vector<int> fixed;for(int i=0;i<k;++i)fixed.push_back(i);
                auto key=canonical(g,fixed);cout<<key.first<<' '<<key.second<<'\n';
            }return 0;
        }
        if(argc!=4)throw runtime_error("usage: atlas MAX_VERTICES MAX_DEGREE OUTPUT_DIR");
        int limit=stoi(argv[1]),cap=stoi(argv[2]);
        if(limit<1||limit>11||cap<1||cap>=11)throw runtime_error("supported vertices 1..11, degree 1..10");
        filesystem::path out(argv[3]);filesystem::create_directories(out);
        vector<vector<Key>> by_n(limit+1);by_n[1]={{1,0}};
        for(int n=2;n<=limit;++n){
            set<Key> current;
            for(auto key:by_n[n-1]) {
                Graph base=decode(key);unsigned available=0;
                for(int u=0;u<n-1;++u)if(base.degree(u)<cap)available|=1u<<u;
                for(unsigned s=available;s;s=(s-1)&available){
                    if(__builtin_popcount(s)>cap)continue;
                    Graph g=base;g.n=n;
                    for(int u=0;u<n-1;++u)if(s&(1u<<u))g.edge(u,n-1);
                    current.insert(canonical(g));
                }
            }
            by_n[n]=vector<Key>(current.begin(),current.end());
            cerr<<"connected hosts n="<<n<<" degree<="<<cap<<": "<<current.size()<<'\n';
        }
        vector<Key> hosts;for(int n=1;n<=limit;++n)hosts.insert(hosts.end(),by_n[n].begin(),by_n[n].end());
        vector<map<Key,int>> histograms;set<Key> all_balls;
        for(auto key:hosts){
            Graph g=decode(key);map<Key,int> h;
            for(int u=0;u<g.n;++u)++h[ball(g,u,2)];
            for(auto& kv:h)all_balls.insert(kv.first);
            histograms.push_back(move(h));
        }
        vector<Key> balls(all_balls.begin(),all_balls.end());map<Key,int> ids;
        ofstream bf(out/"balls.tsv");bf<<"# id vertices adjacency_mask; root=0\n";
        vector<Graph> bg;
        for(int i=0;i<(int)balls.size();++i){ids[balls[i]]=i;bg.push_back(decode(balls[i]));bf<<i<<'\t'<<balls[i].first<<'\t'<<balls[i].second<<'\n';}
        ofstream hf(out/"hosts.tsv");hf<<"# id vertices adjacency_mask\n";
        ofstream hh(out/"histograms.tsv");hh<<"# host_id ball_id:multiplicity ...\n";
        for(int i=0;i<(int)hosts.size();++i){
            hf<<i<<'\t'<<hosts[i].first<<'\t'<<hosts[i].second<<'\n';hh<<i;
            for(auto& kv:histograms[i])hh<<'\t'<<ids.at(kv.first)<<':'<<kv.second;
            hh<<'\n';
        }
        ofstream pf(out/"products.tsv");pf<<"# nonunit factor_a factor_b product; a<=b\n";
        size_t products=0;
        for(int i=1;i<(int)balls.size();++i)for(int j=i;j<(int)balls.size();++j){
            const Graph& a=bg[i];const Graph& b=bg[j];
            if(a.n+b.n>limit+1)break;
            int n=a.n+b.n-1+a.degree(0)*b.degree(0);
            if(n>limit||a.degree(0)+b.degree(0)>cap)continue;
            Graph c=local_product(a,b);
            if(c.n!=n)throw runtime_error("product size identity failed");
            if(c.maxdegree()>cap)continue;
            auto key=canonical(c,{0});
            if(!ids.count(key))throw runtime_error("product missing from complete ball catalog");
            pf<<i<<'\t'<<j<<'\t'<<ids.at(key)<<'\n';++products;
        }
        cerr<<"rooted balls="<<balls.size()<<"; retained nonunit products="<<products<<'\n';
    }catch(const exception& e){cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
