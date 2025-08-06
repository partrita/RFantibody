# RFantibody

## 구조 기반 _de novo_ 항체 디자인

![배너](https://www.bakerlab.org/wp-content/uploads/2025/02/RFdiffusion-antibody-bound-to-Cdiff-ToxinB-BY-Ian-C-Haydon-University-of-Washington-1024x576.jpg)

# 설명

RFantibody는 구조 기반의 _de novo_ 항체 및 나노바디 디자인을 위한 파이프라인입니다. RFantibody는 세 가지 개별적인 방법으로 구성됩니다:
- [RFdiffusion](https://www.nature.com/articles/s41586-023-06415-8)의 항체 미세조정 버전을 이용한 단백질 골격 디자인
- [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187)을 이용한 단백질 서열 디자인
- [RoseTTAFold2](https://www.biorxiv.org/content/10.1101/2023.05.24.542179v1)의 항체 미세조정 버전을 이용한 디자인의 _in silico_ 필터링

RFantibody 파이프라인은 [이 사전 인쇄본](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v1)에 자세히 설명되어 있습니다.

# 목차
- [요구사항](#요구사항)
- [설치](#설치)
- [사용법](#사용법)
  - [HLT 파일 형식](#hlt-파일-형식)
  - [입력 준비](#입력-준비)
  - [RFdiffusion](#rfdiffusion)
  - [ProteinMPNN](#proteinmpnn)
  - [RF2](#rf2)
- [항체 디자인을 위한 실용적인 고려사항](#항체-디자인을-위한-실용적인-고려사항)
  - [타겟 부위 선택](#타겟-부위-선택)
  - [나노바디 도킹](#나노바디-도킹)
  - [타겟 단백질 절단](#타겟-단백질-절단)
  - [핫스팟 선택](#핫스팟-선택)
  - [항체 디자인 규모](#항체-디자인-규모)
  - [CDR 길이 선택](#cdr-길이-선택)
  - [필터링 전략](#필터링-전략)
- [Quiver 파일](#quiver-파일)
- [결론](#결론)


# 요구사항

### Docker

RFantibody는 Docker 컨테이너에서 실행되도록 설계되었습니다. 컨테이너는 호스트 운영 체제 위에 별도의 운영 체제를 실행합니다. 이는 다음과 같은 이점을 제공합니다:
- 간소화된 설치: Docker 소프트웨어 스위트만 있으면 됩니다.
- 호스트 시스템 불변성: 컨테이너 내부에서 실행되기 때문에 어디서 실행하든 본질적으로 동일하게 동작합니다.

호스트 시스템에 설치해야 할 것은 [여기](https://docs.docker.com/engine/install/)에서 무료로 설치할 수 있는 Docker뿐입니다.
클라우드 컴퓨팅에서 RFantibody를 실행하는 경우 Docker가 사전 설치되어 있는 경우가 많습니다. 다음을 실행하여 이를 확인할 수 있습니다:
```bash
which docker
```
이 명령이 경로를 반환하면 Docker를 사용할 수 있으며 준비가 된 것입니다.

### GPU 가속

RFantibody를 실행하려면 NVIDIA GPU가 필요합니다. 다음을 실행하여 사용 가능한 NVIDIA GPU가 있는지 확인할 수 있습니다:
```bash
nvidia-smi
```
이 명령이 성공적으로 실행되면 호환되는 GPU가 있으며 RFantibody가 그 위에서 실행될 수 있습니다.

# 가중치 다운로드

RFantibody가 다운로드된 디렉토리로 이동합니다. 그런 다음 다음 명령을 실행하여 파이프라인 가중치를 RFantibody/weights 디렉토리로 다운로드합니다.
```bash
bash include/download_weights.sh
```

# 설치

## RFantibody Docker 컨테이너 빌드 및 실행

Docker 컨테이너를 시작할 수 있는 올바른 권한이 있는지 확인하려면 다음을 실행해야 합니다:

```bash
sudo usermod -aG docker $USER
```

이 명령을 실행한 후 이 변경 사항을 적용하려면 터미널 세션을 다시 시작해야 합니다.


### Docker 이미지 빌드

RFantibody가 다운로드된 디렉토리로 이동합니다. 그런 다음 다음 명령을 실행하여 RFantibody용 Docker 이미지를 빌드합니다:
```bash
docker build -t rfantibody .
```

### Docker 이미지 시작

방금 빌드한 이미지를 기반으로 Docker 컨테이너를 시작하려면 다음 명령을 실행합니다:
```bash
docker run --name rfantibody --gpus all -v .:/home --memory 10g -it rfantibody
```
이렇게 하면 마지막 명령을 실행한 디렉토리를 미러링하는 /home 디렉토리의 RFantibody 컨테이너로 들어가게 됩니다.



## 파이썬 환경 설정

RFantibody 컨테이너에서 다음을 실행하여 파이썬 환경을 설정합니다:
```bash
bash /home/include/setup.sh
```
이는 다음을 수행합니다:
- 파이썬 환경 빌드 준비를 위해 [Deep Graph Library](https://www.dgl.ai) 다운로드
- [Python Poetry](https://python-poetry.org)를 사용하여 파이썬 환경 빌드
- [USalign](https://github.com/pylelab/USalign) 실행 파일 빌드

## 프로덕션 Docker 이미지

위에 설명된 프로세스는 개발에 적합합니다. 그러나 RFAntibody 파이프라인을 프로덕션 환경에 배포하는 경우 모든 종속성이 사전 설치된 Docker 이미지를 사용하는 것이 좋습니다. `production.Dockerfile` Dockerfile이 이를 수행합니다. 모든 종속성이 설치된 이미지를 빌드합니다. RFAntibody 소스와 스크립트는 `/opt/rfantibody`에 설치됩니다.

### 프로덕션 Docker 이미지 빌드

```bash
docker build -t rfantibody-production -f production.Dockerfile.
```


# 사용법

## HLT 파일 형식

RFantibody 파이프라인의 다른 단계 간에 구조를 전달해야 합니다. 파이프라인의 각 단계는 다음을 알아야 합니다:
- 현재 디자인하고 있는 항체-표적 복합체 구조
- 어느 사슬이 중쇄, 경쇄 및 표적 사슬인지
- 어느 잔기가 어느 CDR 루프에 있는지

파이프라인 단계 간에 이 정보를 전달할 수 있도록 HLT 파일이라는 파일 형식을 정의합니다. HLT 파일은 단순히 .pdb 파일이지만 다음과 같은 수정 사항이 있습니다:
- 중쇄는 사슬 ID 'H'로 표시됩니다.
- 경쇄는 사슬 ID 'L'로 표시됩니다.
- 표적 사슬은 사슬 ID 'T'로 표시됩니다 (여러 표적 사슬이 있는 경우에도).
- 파일의 사슬 순서는 중쇄, 경쇄, 표적 순입니다.
- 파일 끝에는 각 CDR 루프의 1-인덱싱된 절대 (사슬당이 아닌) 잔기 인덱스를 나타내는 PDB Remark가 있습니다. 예:
  ```text
  REMARK PDBinfo-LABEL:   32 H1
  REMARK PDBinfo-LABEL:   52 H2
  ```

## 입력 준비

RFantibody의 항체 미세조정 버전 RFdiffusion은 HLT-remarked 프레임워크 구조를 입력으로 필요로 합니다. 다음과 같이 실행할 수 있는 이 변환을 수행하는 스크립트를 제공합니다:

```bash
# rfantibody 컨테이너 내부에서
poetry run python /home/scripts/util/chothia2HLT.py  \
    -i /home/scripts/examples/rfdiffusion/example_inputs/8tlm_chothia.pdb \
    -o /home/scripts/examples/rfdiffusion/example_inputs/8tlm \
    -H A -L B -T C
```

```bash
# rfantibody 컨테이너 내부에서

bash /home/scripts/examples/generate_HLT_8tlm.sh
```

이 스크립트는 Chothia 주석이 달린 .pdb 파일을 예상합니다. 이러한 파일을 위한 훌륭한 소스는 [SabDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab)이며, PDB의 모든 항체 및 나노바디의 Chothia 주석 구조를 제공하고 몇 달마다 업데이트됩니다.

RFantibody 사전 인쇄본의 디자인 캠페인에서 사용된 HLT 형식의 항체 및 나노바디 프레임워크를 여기에 제공합니다:

- 나노바디 프레임워크: `/scripts/examples/example_inputs/h-NbBCII10.pdb`
- ScFv 프레임워크: `/scripts/examples/example_inputs/hu-4D5-8_Fv.pdb`

## RFdiffusion

RFantibody의 첫 번째 단계는 항체 미세조정 버전의 RFdiffusion을 사용하여 항체-표적 도크를 생성하는 것입니다. 다음은 RFdiffusion을 실행하는 예제 명령입니다:

```bash
# rfantibody 컨테이너 내부에서

poetry run python  /home/rfantibody/scripts/rfdiffusion_inference.py \
  --config-path src/rfantibody/rfdiffusion/config/inference \
  --config-name antibody \
  antibody.target_pdb=/home/scripts/examples/example_inputs/rsv_site3.pdb \
  antibody.framework_pdb=/home/scripts/examples/example_inputs/hu-4D5-8_Fv.pdb \
  inference.ckpt_override_path=/home/weights/RFdiffusion_Ab.pt \
  'ppi.hotspot_res=[T305,T456]' \
  'antibody.design_loops=[L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13]' \
  inference.num_designs=20 \
  inference.output_prefix=/home/scripts/examples/example_outputs/ab_des
```

> 프로덕션 Docker 이미지를 사용하는 경우 `/home/rfantibody/` 명령을 `/opt/rfantibody/`로 변경하십시오.

이러한 구성이 무엇을 하는지 더 자세히 이해하기 위해 이 명령을 살펴보겠습니다:
- antibody.target_pdb: 항체를 디자인하려는 대상 구조의 경로입니다. 이는 파이프라인 실행의 계산 비용을 줄이기 위해 일반적으로 잘린 대상 구조입니다. 자르기 전략은 [여기](#타겟-단백질-절단)에서 더 자세히 설명합니다.
- antibody.framework_pdb: 디자인에 사용하려는 HLT 형식의 항체 프레임워크 경로입니다. RFdiffusion은 루프로 주석이 달린 프레임워크 영역의 구조와 서열만 디자인하므로 이미 최적화된 프레임워크의 도크와 루프를 디자인할 수 있습니다.
- inference.ckpt_override_path: 추론에 사용할 RFdiffusion 모델 가중치 세트의 경로입니다.
- ppi.hotspot_res: 에피토프를 정의하는 핫스팟 잔기 목록입니다. 이것들은 바닐라 RFdiffusion과 동일한 형식으로 제공됩니다. 핫스팟 선택에 대한 자세한 내용은 [여기](#타겟-부위-선택)에서 논의합니다.
- antibody.design_loops: 각 CDR 루프를 허용된 루프 길이 범위에 매핑하는 사전입니다. 각 루프의 길이는 이 범위에서 균일하게 샘플링되며 다른 루프에 대해 샘플링된 길이와 독립적입니다. 프레임워크에 CDR 루프가 있지만 dict에 없는 경우 이 CDR 루프의 서열과 구조는 디자인 중에 고정됩니다. CDR 루프가 dict에 포함되어 있지만 길이 범위가 제공되지 않은 경우 이 CDR 루프는 서열과 구조가 디자인되지만 프레임워크 구조에 제공된 루프 길이로만 디자인됩니다.
- inference.num_designs: 생성해야 할 디자인 수입니다.
- inference.output_prefix: 생성할 .pdb 파일 출력의 접두사입니다.

다음과 같이 실행할 수 있는 예제 입력이 포함된 예제 명령을 제공합니다:

```bash
# rfantibody 컨테이너 내부에서

bash /home/scripts/examples/rfdiffusion/antibody_pdbdesign.sh
```

## ProteinMPNN

RFantibody의 두 번째 단계는 RFdiffusion에 의해 생성된 도크를 가져와 CDR 루프에 서열을 할당하는 것입니다. 우리는 ProteinMPNN의 기본 버전을 사용하여 이 작업을 수행합니다. 즉, 항체 미세조정 모델이 아닙니다. 편의를 위해 이 리포지토리에 필요한 ProteinMPNN 스크립트를 패키징하고 ProteinMPNN을 사용하여 CDR 루프만 디자인할 수 있도록 하는 래퍼 스크립트를 제공합니다.

가장 간단하게, ProteinMPNN은 다음 명령을 사용하여 HLT 형식의 .pdb 파일 디렉토리에서 실행할 수 있습니다:

```bash
# rfantibody 컨테이너 내부에서

poetry run python /home/scripts/proteinmpnn_interface_design.py \
    -pdbdir /path/to/inputdir \
    -outpdbdir /path/to/outputdir
```

이렇게 하면 모든 CDR 루프가 디자인되고 입력 구조당 하나의 서열이 제공됩니다. 실험할 수 있는 더 많은 인수가 있으며 다음을 실행하여 설명됩니다:

```bash
poetry run python /home/scripts/proteinmpnn_interface_design.py --help
```

다음과 같이 실행할 수 있는 예제 입력이 포함된 예제 명령을 제공합니다:

```bash
# rfantibody 컨테이너 내부에서

bash /home/scripts/examples/proteinmpnn/ab_pdb_example.sh
```

## RF2

RFantibody 파이프라인의 마지막 단계는 항체 미세조정 RF2를 사용하여 방금 디자인한 서열의 구조를 예측하는 것입니다. 그런 다음 RF2가 우리가 디자인한 대로 서열이 결합할 것이라고 확신하는지 평가합니다.

가장 간단하게, RF2는 다음 명령을 사용하여 HLT 형식의 .pdb 파일 디렉토리에서 실행할 수 있습니다:

```bash
# rfantibody 컨테이너 내부에서

poetry run python /home/scripts/rf2_predict.py \
    input.pdb_dir=/path/to/inputdir \
    output.pdb_dir=/path/to/outputdir
```

기본적으로 이것은 10번의 재활용 반복과 모델에 제공된 핫스팟의 10%로 실행됩니다. 이러한 하이퍼파라미터의 어떤 조합이 디자인 성공을 가장 잘 예측할지는 아직 모르지만 더 많은 항체 및 나노바디 캠페인에 대한 데이터가 있으면 이러한 값을 조정할 수 있을 것입니다.

다음과 같이 실행할 수 있는 예제 입력이 포함된 예제를 제공합니다:

```bash
# rfantibody 컨테이너 내부에서

bash /home/scripts/examples/rf2/ab_pdb_example.sh
```

# 항체 디자인을 위한 실용적인 고려사항
항체를 디자인하는 것은 _de novo_ 바인더를 디자인하는 것과 유사하지만 개발 초기 단계에 있습니다. 여기에서는 이 파이프라인을 가장 잘 사용하여 실험적으로 작동할 항체를 디자인하는 방법에 대한 조언과 교훈을 공유합니다. 더 많은 항체 디자인 캠페인이 수행되고 모범 사례가 구체화됨에 따라 이 조언 중 일부가 변경될 것으로 예상합니다. 이 두 방법은 많은 유사점을 공유하고 조언이 두 가지 모두에 적용되므로 이러한 섹션 중 일부는 RFdiffusion README의 유사한 섹션에서 채택되었습니다.

## 타겟 부위 선택

표적 단백질의 모든 부위가 항체 디자인에 좋은 후보는 아닙니다. 결합에 매력적인 후보가 되려면 바인더가 상호 작용할 수 있는 >~3개의 소수성 잔기가 있어야 합니다. 하전된 극성 부위에 결합하는 것은 여전히 매우 어렵습니다. 글리칸이 가까이 있는 부위에 결합하는 것도 종종 결합 시 정렬되기 때문에 어렵고 그에 대한 에너지적 손실을 감수해야 합니다. 비구조적 루프에 결합하는 것은 역사적으로 어려웠지만 [이 논문](https://www.nature.com/articles/s41586-023-06953-1)은 RFdiffusion을 사용하여 비구조적 펩타이드를 결합하는 전략을 설명하며, 이는 비구조적 루프와 많은 공통점을 공유합니다. 이 전략을 항체와 함께 사용하면 효과가 있어야 하지만 루프의 유연성에 따라 결합 중에 루프를 정렬하는 데 에너지 비용을 지불해야 합니다.

## 나노바디 도킹

나노바디 출력을 보기 시작하면 많은 것들이 측면 도킹으로 결합하고 있음을 알 수 있습니다. 이것은 버그가 아니며 종종 이러한 측면 도킹 스타일로 결합하고 일부 프레임워크 매개 접촉을 하는 자연적인 나노바디 도크에 대해 모델이 훈련된 결과입니다. 핫스팟과 CDR 길이를 조정하여 더 항체와 유사한 도크를 얻을 수 있지만, 항체와 유사한 도크를 원한다면 항체 프레임워크로 디자인하는 것이 좋습니다.

## 타겟 단백질 절단

RFdiffusion 및 RF2는 시스템의 잔기 수인 N에 대해 O(N^2)으로 런타임이 확장됩니다. 따라서 계산이 불필요하게 비싸지 않도록 큰 타겟을 자르는 것이 매우 좋습니다. RFantibody 파이프라인의 모든 단계는 잘린 타겟을 허용하도록 설계되었습니다. 타겟을 자르는 것은 예술입니다. 다중 도메인 세포외막과 같은 일부 타겟의 경우, 두 도메인이 유연한 링커로 연결된 곳이 자연스러운 절단 지점입니다. 바이러스 스파이크 단백질과 같은 다른 단백질의 경우 이 절단 지점은 덜 명확합니다. 일반적으로 2차 구조를 보존하고 가능한 한 적은 사슬 파손을 도입하려고 합니다. 또한 의도한 타겟 부위의 각 측면에 ~10A의 타겟 단백질을 남겨두려고 노력해야 합니다. PyMol을 사용하여 타겟 단백질을 자르는 것이 좋습니다.

## 핫스팟 선택

핫스팟은 항체가 상호 작용할 타겟의 부위를 제어할 수 있도록 모델에 통합한 기능입니다. 훈련 중에 가장 가까운 5개 항체 CDR 잔기까지의 평균 Cβ 거리가 8옹스트롬 미만인 경우 타겟 잔기를 핫스팟으로 분류합니다. 타겟에서 식별된 모든 핫스팟 중 0-100%가 실제로 모델에 제공되고 나머지는 마스킹됩니다. RFantibody는 바닐라 RFdiffusion보다 어떤 핫스팟이 선택되는지에 더 민감하다는 것을 발견했습니다. RFdiffusion은 잘못된 핫스팟 세트가 주어지면 긴 나선을 생성하는 경향이 있는 반면, RFantibody는 잘못된 핫스팟 세트가 주어지면 일반적으로 도킹되지 않은 항체를 생성합니다. 수천 개의 디자인을 생성하기 전에 몇 번의 파일럿 실행을 실행하여 제공하는 핫스팟 수가 원하는 결과를 제공하는지 확인하는 것이 매우 좋습니다.

## 항체 디자인 규모

우리 원고에서 보고하는 일부 타겟 캠페인의 경우 95개 디자인 세트에서 VHH 바인더를 식별할 수 있었습니다. 그러나 더 일반적인 경우, 히트를 식별하려면 10k 범위의 디자인 캠페인이 필요할 것으로 예상합니다. 이것은 신뢰할 수 있는 필터링 메트릭이 없기 때문입니다([필터링 전략](#필터링-전략) 섹션에서 자세히 설명). 긍정적인 데이터와 부정적인 데이터 모두 필터를 조정하고 평가하는 데 유용하므로 디자인 캠페인을 실행하고 데이터를 더 넓은 커뮤니티와 공유하고자 한다면 더 신뢰할 수 있는 필터, 더 높은 성공률 및 더 저렴한 디자인 캠페인으로 나아가는 데 매우 도움이 될 것입니다.

## CDR 길이 선택

디자인 캠페인에 사용한 루프 범위는 RFdiffusion 예제 파일에 제공됩니다. 우리는 각 루프에 대해 자연적으로 발생하는 길이의 빈도를 보고 우리 범위로 밀도의 대부분을 덮으려고 노력하여 이러한 범위를 결정했습니다. 우리는 또한 상대적으로 짧은 H3 루프를 선택하려고 노력했습니다. 왜냐하면 이것이 효과적으로 결합하기에 충분한 길이를 제공하면서 디자인하고 예측하기가 더 쉬울 것이라고 생각했기 때문입니다. 단백질의 소수성 포켓을 타겟팅하는 경우와 같이 긴 H3가 유용할 수 있는 일부 타겟이 있습니다. 이러한 경우 H3 범위는 예제에서 제공하는 것 이상으로 늘려야 합니다.

## 필터링 전략

다음과 같은 최소 필터링 기준을 권장합니다:
- RF2 pAE < 10
- RMSD (디자인 대 RF2 예측) < 2Å
- Rosetta ddG < -20으로 필터링하는 것도 도움이 될 수 있습니다.

효과적인 필터가 없다는 것이 현재 RFantibody 파이프라인의 주요 한계입니다. 우리가 제공하는 RF2 버전은 일부 경우에 비결합체에 비해 결합체의 약한 농축을 보일 수 있지만 이 결론을 설득력 있게 내리려면 더 많은 데이터가 필요합니다. AF3과 같은 새로 사용 가능한 구조 예측 모델은 RF2에 대한 유망한 대안을 제시하며 우리는 디자인 캠페인에 대한 예측 가능성에 대해 이러한 모델을 평가하는 과정에 있습니다.

# Quiver 파일

대규모 디자인 캠페인을 실행할 때 많은 디자인과 해당 디자인과 관련된 점수를 보유하는 단일 파일을 갖는 것이 종종 유용합니다. 이것은 수천 개의 개별 .pdb 파일을 저장하고 액세스하는 것보다 파일 시스템에 더 부드럽습니다. RFantibody 파이프라인에서 [Quiver 파일](https://github.com/nrbennet/quiver)을 사용할 수 있는 기능을 제공합니다. 이 파일은 단순히 많은 작은 파일의 내용이 들어 있는 하나의 큰 파일입니다. 각 항목에는 고유한 이름이 있으며 항목에 대한 메타데이터를 저장할 수 있습니다.

이 리포지토리에는 구성 가능한(파이프 가능한) 명령으로 Quiver 파일을 조작할 수 있는 여러 명령줄 도구도 있습니다.

Quiver 파일과 다양한 Quiver 도구는 Brian Coventry의 [silent_tools](https://github.com/bcov77/silent_tools) 프로젝트에서 크게 영감을 받았습니다. 차이점은 Quiver 파일이 Rosetta 외부 환경에서 작동할 수 있다는 점이며 이는 매우 편리합니다. Quiver 파일 명령줄 도구는 silent_tools의 직접적인 유사체이며 silent_tools를 사용해 본 사람들에게는 익숙할 것입니다:

```bash
# quiver 파일 만들기
qvfrompdbs *.pdb > my.qv

# quiver 파일에 무엇이 있는지 묻기
qvls my.qv

# quiver 파일에 몇 개가 있는지 묻기
qvls my.qv | wc -l

# quiver 파일에서 모든 pdb 추출하기
qvextract my.qv

# quiver 파일에서 처음 10개 pdb 추출하기
qvls my.qv | head -n 10 | qvextractspecific my.qv

# quiver 파일에서 무작위로 10개 pdb 추출하기
qvls my.qv | shuf | head -n 10 | qvextractspecific my.qv

# quiver 파일에서 특정 pdb 추출하기
qvextractspecific my.qv name_of_pdb_0001

# quiver 파일에서 스코어 파일 생성하기
qvscorefile my.qv

# qv 파일 결합하기
cat 1.qv 2.qv 3.qv > my.qv

# quiver 파일의 모든 pdb에 고유한 이름이 있는지 확인하기
qvls my.qv | qvrename my.qv > uniq.qv

# quiver 파일을 100개 그룹으로 분할하기
qvsplit my.qv 100
```

## Quiver 파일 읽기 및 쓰기
RFantibody의 모든 단계에서 Quiver 파일을 사용할 수 있습니다. 구문은 여기에 요약되어 있습니다:

RFdiffusion은 `.pdb` 파일 타겟과 프레임워크만 입력으로 받습니다. 디자인된 백본을 Quiver 파일로 출력하려면 입력 명령에 이 인수를 추가하십시오:

```bash
inference.quiver=/path/to/myoutput.qv
```

ProteinMPNN의 경우 Quiver 파일을 입력 및 출력하려면 다음 두 인수를 사용하십시오:

```bash
-inquiver /path/to/myinput.qv -outquiver /path/to/myoutput.qv
```

RFantibody는 Quiver 파일 입력 및 출력과 함께 작동하기 위해 다음 두 가지 구성을 사용합니다.

```bash
input.quiver=/path/to/myinput.qv output.quiver=/path/to/myoutput.qv
```

# 결론

RFantibody를 오픈 소스로 출시하게 되어 정말 기쁩니다! 더 넓은 커뮤니티가 어떤 종류의 디자인을 내놓을지 기대됩니다. 이 코드베이스를 가능한 한 쉽게 설정하고 실행할 수 있도록 열심히 노력했지만 문제가 발생하면 GitHub 문제를 열어주세요.- Nate, Joe 및 RFantibody 팀

---

RFantibody는 우리가 여기서 인정하는 여러 방법의 아키텍처와 가중치를 직접 기반으로 합니다. 원래 RFdiffusion 및 항체 미세 조정 RoseTTAFold2 모델의 기반이 된 RoseTTAFold 및 RoseTTAFold2를 개발한 Minkyung Baek 및 Frank DiMaio에게 감사드립니다. 이 리포지토리에서 항체별 래퍼를 제공하는 ProteinMPNN을 개발한 Justas Dauparas에게 감사드립니다. 여기서 제공하는 항체 미세 조정 RFdiffusion은 원래 버전의 RFdiffusion을 직접 기반으로 하므로 우리와 함께 원래 RFdiffusion을 공동 개발한 David Juergens, Brian Trippe 및 Jason Yim에게도 감사드립니다. RFantibody는 MIT 라이선스(LICENSE 파일 참조)에 따라 출시됩니다. 비영리 및 영리 목적 모두 무료입니다.
