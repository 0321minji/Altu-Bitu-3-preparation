#include<iostream>
#include<vector>
#include<cmath>
#include<set>
using namespace std;

void isPrime(int n, vector<bool> &is_prime) {//�����佺�׳׽��� ü 

	is_prime[0] = is_prime[1] = false;

	for (int i = 2; i <= sqrt(n); i++) {
		if (is_prime[i]) {
			for (int j = i * i; j <= n; j += i) {
				is_prime[j] = false;
			}
		}
	}
}


int calSumNum(int n) {//�� �ڸ����� ������ ���� �� ���ϴ� �Լ�
	int sum = 0;
	while (n) {
		sum += pow(n % 10, 2);
		n /= 10;
	}
	return sum;
}


bool isSangguen(int n) {

	set<int> s;//������ ���ڸ� key�� Ž���� �� �ִ� set�� ��� (vs �迭�� �ε��� �����̱⿡ ����� ���ڸ� key�� Ž���ϱ� �������)
	int tmp = n;
	while (1) {

		tmp = calSumNum(tmp); //�� �ڸ��� ������ �� ���ϱ�

		if (tmp == 1) {// ��ټ��̸�
			return true;
		}
		if (s.find(tmp) != s.end()) { //��ټ��� �ƴϸ�
			return false;
		}
		s.insert(tmp);
	}
}


vector<int> Solution(int n) {

	vector<bool> is_prime(n + 1, true);
	vector<int> answer;
	isPrime(n, is_prime);

	for (int i = 2; i < n; i ++ ) {
		if (!is_prime[i]) continue; //�Ҽ� �Ǵ�
		else{ 
			if (!isSangguen(i)) {//��ټ� �Ǵ�
				continue;
			}
			else {
				answer.push_back(i);
			}
		}
	}
	return answer;
}

/*
* isPrime�Լ��� �Ҽ����� �Ǵ��ϰ�
* isSangguen�Լ��� ��ټ����� �Ǵ��ؿ�. ��ټ� �Ǵ� �ÿ��� ��ټ��� ���� + ��ټ��� �ƴ� ������ �������ּ���
*/

int main() {
	int n;

	cin >> n;

	 vector<int> v = Solution(n);

	for (int i = 0; i <v.size(); i++) {
		cout << v[i] << '\n';
	}

	return 0;
}