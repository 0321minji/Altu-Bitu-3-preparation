#include <iostream>
#include <vector>
#include <tuple>
#include <queue>

using namespace std;
typedef tuple<int, int, int> tp;

vector<int> parent;

//Find ����
int findParent(int node) {
    if (parent[node] < 0) //���� ������ ��Ʈ ����
        return node;
    return parent[node] = findParent(parent[node]); //�׷��� �����ϸ� ��Ʈ ���� ã��
}

//Union ����
bool unionInput(int x, int y) {
    int xp = findParent(x);
    int yp = findParent(y);

    if (xp == yp) //���� ���տ� �ִٸ� ���Ͽ� �� �� ����
        return false;
    if (parent[xp] < parent[yp]) { //���ο� ��Ʈ xp
        parent[xp] += parent[yp];
        parent[yp] = xp;
    }
    else { //���ο� ��Ʈ yp
        parent[yp] += parent[xp];
        parent[xp] = yp;
    }
    return true;
}

long long kruskal(int v, long long tot, priority_queue<tp, vector<tp>, greater<>>& pq) {
    int cnt = 0;
    long long sum = 0;

    while (cnt < v - 1) { //����� ������ ���� v-1���� ���� ����
        if (pq.empty()) //����� ������ v-1���� �ȵƴµ� �� �̻� �˻��� ������ ���ٸ�
            return -1;

        int cost = get<0>(pq.top());
        int x = get<1>(pq.top());
        int y = get<2>(pq.top());

        pq.pop();
        if (unionInput(x, y)) { //���Ͽ� �ߴٸ�
            cnt++;
            sum += cost;
        }
    }
    return tot - sum;
}

/**
 * �⺻ MST �������� Ʈ���� ���� �� ���� ���(������ N-1���� �ƴ� ���)�� ����� ����
 *
 * �ִ� ����� ������ �� 10^6 x 10^5 �̹Ƿ� long long �ڷ����� ��� ��
 */

int main() {
    int n, m, a, b, c;
    long long tot = 0;
    priority_queue<tp, vector<tp>, greater<>> pq;

    //�Է�
    cin >> n >> m;
    parent.assign(n + 1, -1);
    while (m--) {
        cin >> a >> b >> c;
        pq.push({ c, a, b });
        tot += c; //���θ� �� ��ġ�� �� ��� ���
    }

    //���� & ���
    cout << kruskal(n, tot, pq);
}