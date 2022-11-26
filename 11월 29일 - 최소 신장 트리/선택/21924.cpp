#include<iostream>
#include<vector>
#include<queue>

using namespace std;
typedef pair<int, int> pi;
const int INF =1e6+1;

long long prim(int n,vector<vector<pi>> graph) {
	priority_queue<pi> pq;
	vector<int> cost(n + 1, INF); //cost
	vector<bool>visit(n + 1, false); //�湮 check

	long long sum=0;
	//�ʱ�ȭ
	cost[1] = 0;
	pq.push(pi(0, 1));

	while (!pq.empty()) {
		
		int cur = pq.top().second;
		pq.pop();

		if (visit[cur]) {
			continue;
		}
		visit[cur] = true;
		
		for (auto i : graph[cur]) {
			if (!visit[i.second] && (cost[i.second] > i.first)) {
				cost[i.second] = i.first;
				pq.push(make_pair(-cost[i.second], i.second));
			}
		}

	}
	for (int i = 1; i <= n; i++) {
		if (cost[i] == INF) {
			return -1;
		}
		sum += cost[i];
	}
	return sum;
}

/*
������ ����� ���ؾ��ϹǷ� ó���� �� ��� ���ؾ���
pair �� ���� graph�� ���� �� prim���� MST ���� �� ���ϱ�
*/


int main() {
	int n, m;
	cin >> n >> m;
	vector<vector<pi>> graph = vector<vector<pi>>(n + 1);

	int a, b, c;
	long long price = 0;

	for (int i = 0; i < m; i++) {
		cin >> a >> b >> c;
		graph[a].push_back(make_pair(c, b));
		graph[b].push_back(make_pair(c,a));
		price += c;
	}
	
	long long result = prim(n, graph);
	//������ �ݾ� ����̹Ƿ� price���� MST��� ���ֱ�
	printf("%lld", result == -1 ? -1 :price- result);
}