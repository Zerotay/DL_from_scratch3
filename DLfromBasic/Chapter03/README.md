## step 25: 계산그래프 시각화(1)
```
# 그래프비즈 설치
sudo apt-get update
sudo apt-get install graphviz
```
그래프 시각화 툴로 이 툴의 graph 문법을 활용해 그림을 그릴 수 있음.

```
# .dot 파일
graph {
    1 [label="x", color=orange, style=filled, shape=box]
    2 ...
    1 -> 2
}
```
위와 같은 형태로 작성하며 숫자와 대괄호 내 정보를 통해 노드를 생성, 그 후 밑에 화살표 자판으로 연결하면 엣지를 그린다.

엣지 꺽쇠에 따라 단방향, 양방향 등 지정 가능하다.

```
# 그래프비즈 실행
dot step25.dot -T png -o sample.png 
```

## step 26: 계산그래프 시각화(2)

계산그래프 그리는 함수를 dezero.utils 에 추가하여 어떤 변수든 그 과정을 추적한 그래프를 쉽게 그릴 수 있도록 하였다.

plot_dot_graph 함수에 Variable 객체를 넣으면 생성 과정에서 거친 모든 함수와 변수들의 그래프를 그린다.

- os.path.splitext(경로) 사용 시, split과 동일하다.
- subprocess 기본 패키지를 임포트하고 subprocess.run(실제 커맨드(str), shell=True) 코드 작성 시 실제 cmd 실행이 일어난다.

https://pypi.org/project/graphviz/

그래프비즈는 파이썬 패키지로도 제공되며 보다 편리하게 사용할 수 있다.

## step 27: 테일러 급수 미분
sin 함수 미분 -> -cos, -sin, cos, sin, -cos ....
해석적으로 sin 함수의 n계 도함수는 위와 같이 표현할 수 있다. 이를 통해 [테일러 급수 매클로린 전개](https://ko.wikipedia.org/wiki/%ED%85%8C%EC%9D%BC%EB%9F%AC_%EA%B8%89%EC%88%98)를 사용할 수 있다. 

테일러 급수는 복잡한 수식을 가진 함수를 다항식으로 표현 가능하다. 다항식으로 표현할 수 있다면 프로그래밍 입장에서 반복문과 기본적인 사칙연산을 통해 근사하는 함수값을 구할 수 있다.

**질문**
- np.cos은 어떻게 구현이 되어있는가?

## step 28: 함수 최적화
[로젠브록 함수](https://ko.wikipedia.org/wiki/%EB%A1%9C%EC%A0%A0%EB%B8%8C%EB%A1%9D_%ED%95%A8%EC%88%98)를 이용하여 등고선을 그리고 가장 출력값이 낮은 지점을 찾는다. 
함수 최적화 문제에서 최대, 최소 지점을 얼마나 빠르고 정확하게 찾는지 비교분석(벤치마크) 할 때 사용하는 함수 중 하나이다.

책에서 경사하강법을 통해 로젠브록 함수의 최소값을 찾아나간다. 이전 스텝에서 구현한 cleargrad 멤버함수를 이용해 매 반복마다 그라디언트를 0으로 두고, 새로운 그라디언트를 구한다.
