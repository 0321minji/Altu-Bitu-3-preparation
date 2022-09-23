#include <iostream>
#include <queue>
using namespace std;

/*
* ���̵��� ������ ���� ���� ����ִ� ���ڿ��� ������ ������
* ������ ������ �������� ������ �켱������ ����...!
* �Ǹ��ϴ� ���� -> ���ڿ� ���ϴ� �������� ���� ������ ����ִ�...!
* 
* Hint ���̵��� ������ ���ڴ� � Ư¡�� �ֳ���?
*/


int solution(int n, int m) {

	int tmp, child_want, max;
	priority_queue<int> max_heap;

	//�켱����ť�� ���� ����
	for (int i = 0; i < n; i++) {
		cin >> tmp;
		max_heap.push(tmp);
	}

	//���̵��� ���ϴ� ������ �Է¹ް� �켱������ ���� ���ڿ� �� �������� ���� ������ ����ִٸ� return 0
	//�켱������ ���� ���� pop�ؼ� ���̰� ������ ���� ���� �ٽ� ť�� ���� -> ������ �켱���� ����
	for (int i = 0; i < m; i++) {

		cin >> child_want;
		if (max_heap.top() < child_want) {
			return 0;
		}
		else {
			max = max_heap.top();
			max_heap.pop();
			max_heap.push(max - child_want);
		}
	}
	return 1;
}


int main() {
	int n, m;
	cin >> n >> m;

	cout << solution(n, m);

	return 0;
}