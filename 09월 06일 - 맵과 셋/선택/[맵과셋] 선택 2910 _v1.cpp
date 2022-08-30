#include <iostream>
#include <map>
#include <algorithm>
#include <vector>

using namespace std;

map<int, int> order;
bool cmp(pair<int, int>& a, pair<int, int>& b) {
	if (a.second == b.second) {//�󵵰� ���ٸ�
		if (order[a.first] < order[b.first])  return 1;//a�� �� ���� ��������
		else return 0;
	}
	else {//�󵵰� �ٸ��ٸ�
		if (a.second > b.second) return 1;
		else return 0;
	}

	}

int main() {

	int N, C, i, num;
	map<int, int> frequency;
	
	

	cin >> N >> C;

	//�Է¹��� ������ frequency�� ������ �� +1 ������ order ����ϰ� freq+1
	for (i = 0; i < N; i++) {
		cin >> num;

		if (!frequency[num]) {
			order[num] = i;
			frequency[num] = 1;
		}
		else {
			frequency[num]++;
		}
	}

	//�󵵰� ���� ���� -> �󵵰� ���ٸ� order���� ����....
	vector<pair<int, int>> v(frequency.begin(), frequency.end());
	sort(v.begin(), v.end(), cmp);

	for (i =0 ; i < v.size(); i++) {
		while (v[i].second) {
			cout << v[i].first << ' ';
			v[i].second--;
		}
	}

	return 0;
}