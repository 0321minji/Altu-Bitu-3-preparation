#include<iostream>
#include<vector>
#include<queue>

using namespace std;
typedef pair<int, int> pi;
const int INF = 1e6 + 1;

long long prim(int n, vector<vector<pi>>& graph, int start) {
	priority_queue<pi, vector<pi>, greater<>> pq;
	vector<int> cost(n + 1, INF); //cost
	vector<bool>visit(n + 1, false); //�湮 check
	int cnt = 0; //visit�� ���� ��
	long long sum = 0;

	//�ʱ�ȭ
	cost[start] = 0;
	pq.push({ 0, start });

	while (!pq.empty()) {
		int weight = pq.top().first;
		int cur = pq.top().second;
		pq.pop();

		if (visit[cur]) {
			continue;
		}
		visit[cur] = true;
		cnt++;
		sum += weight;

		for (auto [next_w, next] : graph[cur]) {
			if (!visit[next] && (cost[next] > next_w)) {
				cost[next] = next_w;
				pq.push({ next_w, next });
			}
		}

	}
	if (cnt != n) {
		return -1;
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

	while (m--) {
		cin >> a >> b >> c;
		graph[a].push_back({ c, b });
		graph[b].push_back({ c, a });
		price += c;
	}

	long long result = prim(n, graph, 1);
	//������ �ݾ� ����̹Ƿ� price���� MST��� ���ֱ�
	printf("%lld", result == -1 ? -1 : price - result);
}