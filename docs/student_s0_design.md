# Student S0 설계안 — (Text, State) → Z

기준 데이터: `results/step_dataset_v2` (124 rollout = 31 s₀ × {20, 25, 30, 35} cm, 각 240~248 tick, `token_k` (T,64) int8, `state_history/*`, `state_start/*`, `k_stand`). Teacher 내부값(λ, cut, search_history)은 Student 입·출력에 쓰지 않는다.

## 1. 전체 구조

```
text ─ Qwen3-0.6B (frozen, bf16) ─ H_L (N×1024) ─ Linear+LN ─ H'_L (N×512)
                                                         │ cross-attn (key mask)
   Q_skill (16×512, learnable) ─ Skill Decoder ×2 ───────┴─ C (16×512) ─ attn-pool ─ d̂ (aux)

state_t-9..t (10 frames × 151) ─ frame MLP ─ + pos(10) ─ Temporal Enc ×2 ─ S (10×512)

   Q_future (40×512, learnable) ─ Motion Transformer ×4 ─ cross-attn to [C ; S] (26×512, +type emb) ─ M (40×512)
                                                                                                     │
                                                                     Action Head Linear 512→2048 ─ A (40×64×32) ─ argmax ─ k̂ ─ Z = k̂/16
Z_t = Ẑ[0] → frozen SONIC decoder (ReplaceToken hook) → q* → PD → sim/robot → next state → repeat (receding horizon)
```

## 2. 모듈별 텐서

| 모듈 | 입력 | 출력 | 비고 |
|---|---|---|---|
| Qwen3-0.6B | token ids (N ≤ 32) | H_L (N×1024) | frozen, 마지막 hidden state. 텍스트 종류가 적으므로 H_L을 디스크 캐시(학습 중 Qwen 실행 없음) |
| Lang projection | (N×1024) | H'_L (N×512) | Linear + LayerNorm + dropout 0.1 |
| Skill Decoder | Q_skill (16×512), H'_L, key mask | C (16×512) | 2 layers: self-attn(Q) → cross-attn(H'_L) → FFN 2048, pre-LN, 8 heads |
| Magnitude head (aux) | C | d̂ (1) | attention pooling: learnable q_pool (1×512)로 C에 attention → c_pool (512) → MLP 512→256→1. c_pool은 paraphrase loss에도 재사용 |
| State features | rollout/sim state | x_t (151) | §3 |
| State Encoder | X (10×151) | S (10×512) | frame MLP 151→512→512 (LN, GELU) + learned temporal pos emb (10×512) → Transformer encoder 2 layers (8 heads, FFN 2048) |
| Memory | C, S | mem (26×512) | concat + learned type embedding (skill/state 2종) |
| Motion Transformer | Q_future (40×512)+pos, mem | M (40×512) | decoder 4 layers: self-attn(Q, non-causal) → cross-attn(mem) → FFN 2048 |
| Action head | M (40×512) | A (40×64×32) | Linear 512→2048 → view(40, 64, 32) |

배치 차원 B가 모두 앞에 붙는다. 6 layers 대신 4 layers를 S0로 제안: 샘플이 124 rollout에서 나와 중복이 크고(§5), 6층(≈19M)은 과적합 위험만 늘린다. 4층(≈12.6M)으로 시작하고 6층은 ablation으로 둔다. d_model 512 / 8 heads / FFN 2048은 유지.

## 3. State feature (프레임당 151-D, 모두 실제 G1에서 관측 가능)

좌표계: F0 = primitive 시작 tick의 base heading frame(`task.LocalFrame`과 동일 정의, R_start). 월드 pose는 쓰지 않는다.

| 블록 | dim | 정의 |
|---|---|---|
| joint_pos | 29 | q − DEFAULT_ANGLES (MuJoCo order, rollout과 동일) |
| joint_vel | 29 | dq |
| base_lin_vel_F0 | 3 | R_startᵀ v_base |
| base_ang_vel | 3 | body frame (IMU gyro) |
| gravity_body | 3 | gravity_in_body_frame(base_quat) |
| heading_rel | 2 | [cos, sin](yaw_t − yaw_start) |
| feet_rel_base_F0 | 6 | R_startᵀ(p_L − p_base), R_startᵀ(p_R − p_base) |
| foot_contact | 2 | clip(F_L, F_R, 0, 500)/100 |
| progress_base | 3 | R_startᵀ(p_base(t) − p_base(0)) |
| progress_feet | 6 | R_startᵀ(p_L(t) − p_L(0)), R_startᵀ(p_R(t) − p_R(0)) |
| prev_token | 64 | 직전 tick에 실행된 k_{t−1}/16 (t=0이면 k_stand/16) |
| phase | 1 | t·0.02 s (primitive 시작 후 경과 시간) |

합 151. d_cmd − Δx 같은 목표 오차는 넣지 않는다. 정규화(평균/표준편차)는 train split에서만 계산하고 prev_token과 heading_rel, contact는 정규화하지 않는다. 히스토리 T_s = 10 (200 ms). t < 9인 프레임은 state_start로 채운다(snapshot은 정지 상태이므로 물리적으로 맞다).

tick 정렬: rollout에서 `state_history[i]`는 token_k[i]를 실행한 뒤의 상태다. 따라서 tick t의 입력 히스토리는 state_start(t=0) 및 state_history[t−10..t−1], 목표는 token_k[t..t+39].

## 4. Action head와 CE

- A = Linear(M) → (B, 40, 2048) → view (B, 40, 64, 32).
- target y = k + 16 ∈ {0..31}, (B, 40, 64) long.
- L_action = CE(A.view(−1, 32), y.view(−1)), reduction mean → 셀(B·40·64)당 평균. 초기값 ≈ ln 32 = 3.47, 스케일 O(1).
- label smoothing 0.05. 선택 항: ordinal 보조 L_ord = Huber(E_softmax[k]/16, k/16) 가중 0.1 (크기 순서를 살림).
- 추론: 셀별 argmax → k̂ → Z = k̂/16 (항상 grid 위). 관측된 k 범위는 −6..7이라 32 클래스 중 14개만 쓰인다. 클래스 불균형(standing 토큰 지배)은 §5 샘플링으로 다룬다.

## 5. 샘플 생성

rollout 하나(T = 240~248)에서 tick t ∈ [0, T−1]마다 샘플:
- text: 해당 d_cmd의 paraphrase 중 train 집합에서 하나를 무작위로 선택(에폭마다 다르게).
- X_t: §3, 10 프레임.
- 목표: token_k[t..t+39]; T를 넘는 부분은 k_stand로 padding(물리적으로 정확: 마지막 standing 토큰이 계속 유지되므로 "mask"가 아니라 유효 레이블로 취급).
- 구간: motion = [0, n_window + 20) (창 + 0.4 s의 정지 전이, 110~118 tick), standing = 나머지(~130 tick). standing 샘플의 목표는 거의 전부 k_stand라 CE가 자명하다.
- 샘플링 비율: 배치의 70%를 motion 구간, 30%를 standing 구간에서 뽑는다. standing 구간 안에서는 앞 50 tick(완료 직후, "멈춰 서기"를 배우는 구간)을 2:1로 가중한다. 에폭 크기는 rollout 수 × 120으로 정의(모든 tick을 균일 반복하지 않음).
- 유효 샘플 수: train 21 rollout-snapshot × 3 거리 = 63 rollout ≈ 7.5k motion tick + 8.2k standing tick. 작다. 중복이 크므로 dropout 0.1, weight decay 0.05, state 노이즈 증강(q/dq/vel에 N(0, 0.02σ))을 기본으로 둔다.

## 6. Split (snapshot 단위, 누수 방지)

- 거리: train {20, 25, 35}, unseen test 30 (모든 split에서 30 cm rollout은 학습·검증에 쓰지 않음).
- snapshot: 31개를 yaw 순으로 정렬 후 균등 간격으로 7개를 held-out, 나머지 24개 중 3개를 validation(학습 중 모델 선택용), 21개 train. 제안 held-out: s0_02, s0_07, s0_12, s0_16, s0_21, s0_25, s0_29(yaw −49°…+37°, 큰 |yaw|와 non-converged 사례 포함). validation: s0_05, s0_14, s0_24.
- paraphrase: train 6 = EN 0,1,2,4 + KO 5,6; held-out 2 = "Walk forward {cm} cm and stop." (EN) + "{cm}cm 앞으로 걸어가." (KO).
- 정규화 통계는 train 21 snapshot × 3 거리에서만.

평가 bucket (closed-loop):

| # | snapshot | 거리 | paraphrase |
|---|---|---|---|
| 1 | train 21 | 20/25/35 | train 6 |
| 2 | train 21 | 30 | train 6 |
| 3 | held-out 7 | 20/25/35 | train 6 |
| 4 | held-out 7 | 30 | train 6 |
| 5 | held-out 7 | 20/25/30/35 | held-out 2 |

## 7. Loss

L = L_action + 0.1·L_mag + 0.05·L_para (+ 0.1·L_ord 선택)

- L_action: 셀 평균 CE, O(1~3). 40×64 셀 평균이므로 가중은 샘플 수에 무관.
- L_mag: Huber(d̂, d_cmd/0.1), δ = 0.5. 단위를 0.1 m(dm)로 두어 목표가 2.0~3.5가 되게 한다(미터 단위면 손실이 10⁻³ 수준이라 0.1 가중으로는 작동하지 않음). d_cmd는 명령값(실측 아님).
- L_para: 1 − cos(c_pool(text_a), c_pool(text_b)), 같은 d_cmd의 서로 다른 paraphrase 쌍(배치 안에서 양성 쌍 구성). 범위 [0, 2]. 음성 쌍(다른 거리)을 밀어내는 항은 S0에서 넣지 않는다(d̂ head가 이미 분리 압력을 준다).
- 가중 검토: 초기 CE 3.5 대비 0.1·L_mag ≈ 0.1~0.2, 0.05·L_para ≈ 0.05로 보조항이 주손실의 5% 이하. 학습 후반 CE가 0.3 이하로 떨어지면 보조항 비중이 커지지만 여전히 CE 우세. 적절하다고 보되, motion 구간 CE를 standing과 분리해 로그로 감시한다.

## 8. 추론(receding horizon)과 closed-loop 평가

루프(20 ms마다): sim state → §3 feature(F0는 primitive 시작 시 고정) → S; C는 명령당 1회 계산 후 캐시 → M → A → argmax → k̂[0] → z = k̂[0]/16 → `ReplaceToken` 훅(needs_base=False, encoder 생략) → SONIC decoder → PD → step. 실행한 k̂[0]은 다음 tick의 prev_token이 된다. 종료: 244 tick 고정(teacher와 동일) 또는 base 속도 < 0.02 m/s가 1 s 지속.

평가 지표(teacher와 동일 코드로 측정): commanded d, realized dx, |오차|, dy, dyaw, fall, end speed, swing 수, offline token accuracy/CE(motion 구간·standing 구간 분리). 기준: bucket 4에서 unseen 30 cm의 realized 30 ± 2 cm, 낙상 0, 두 발 재정렬(gap < 5 cm).

Text conditioning 검증: 같은 s₀에서 "Move forward 20 cm." vs "35 cm."를 넣어 창 구간 k̂ 차이(cell %)와 realized dx 차이(기대 > 10 cm) 측정.

Ablation(같은 학습 모델에 입력만 바꿔 closed-loop):
- correct text / shuffled text(다른 거리의 텍스트로 교체) / blank text(빈 문자열) / no state(S를 train 평균 S로 고정).
- 추가로 state 없이 학습한 모델, prev_token 없이 학습한 모델 두 개를 별도 학습해 비교(prev_token 복사 의존 여부).

## 9. 파일·클래스

새로 만든다 (`src/token_policy/student/`):
- `features.py`: `StateFeaturizer(norm_stats)` — rollout 배열 또는 sim state dict → 151-D, `LocalFrame` 재사용.
- `text_encoder.py`: `QwenTextEncoder` (frozen, bf16) + `TextCache` (text → H_L, mask 저장/로드).
- `dataset.py`: `StepDataset(index, split, sampler cfg)` — 윈도우 샘플, padding, bucket 샘플링, split 생성(`make_split()` → `split.json`).
- `model.py`: `SkillDecoder`, `StateEncoder`, `MotionTransformer`, `ActionHead`, `StudentS0`(forward → A, d̂, c_pool).
- `losses.py`: `action_ce`, `magnitude_huber`, `paraphrase_cos`.
- `policy.py`: `StudentTokenSource` — token_hook 인터페이스(needs_base=False), receding-horizon 상태 관리.
- `experiments/student_train.py`, `experiments/student_eval.py`(bucket별 closed-loop, ablation 옵션), `experiments/student_text_cache.py`.

재사용: `env.StandingSnapshot.restore`(결정론적 reset), `sim.SyncSim/G1Sim`, `sonic.SonicJointController` + `policy.ReplaceToken`, `policy.fsq.k_to_z/z_to_k/snap`, `task.LocalFrame`, `common.rotations.gravity_in_body_frame/calc_heading`, `teacher.step_forward_teacher`의 rollout 측정부(`_heading_frame`, `_swings`, STATE_KEYS → 공용 모듈로 분리 권장), `experiments/step_dataset_view.py`(Student rollout 시각화에 그대로 사용 가능).

설치 필요: `transformers`, `accelerate`(선택), Qwen3-0.6B 가중치(~1.2 GB). pyproject에 `student` extra로 추가.

## 10. 파라미터·VRAM

| 모듈 | 파라미터 |
|---|---|
| Qwen3-0.6B | 0.6B frozen (캐시 후 학습 시 로드 안 함) |
| Lang projection | 0.53M |
| Skill Decoder ×2 | ≈6.3M (+ 8k queries) |
| State Encoder | MLP 0.34M + encoder ×2 ≈ 6.3M |
| Motion Transformer ×4 | ≈12.6M (6층이면 18.9M) |
| Action head | 1.05M |
| Magnitude head + pool | ≈0.4M |
| 학습 파라미터 합 | ≈27M (4층) / ≈34M (6층) |

VRAM: 배치 256, fp32/AMP 기준 활성화 < 3 GB, 전체 < 5 GB. 텍스트 캐시 생성 시 Qwen bf16 ≈ 1.5 GB. 4090 한 장으로 충분. 에폭(≈15k 샘플)당 수십 초, 100 에폭 1시간 이내 예상.

## 11. 위험·애매한 점

1. 언어 다양성이 극히 작다: 학습 텍스트가 3 거리 × 6 paraphrase = 18 문자열. "30 cm" 일반화는 사실상 Qwen 숫자 임베딩의 보간에 의존한다. 보완: 템플릿을 20개 이상으로 늘리고 숫자 표기 변형("25 cm", "25cm", "0.25 m", "twenty-five centimeters")을 추가하되 30은 모든 표기에서 제외. 그래도 S0는 "interpolation 가능성 검증"이지 일반 언어 grounding은 아니다.
2. standing 지배와 자명한 CE: 전체 셀의 60% 이상이 k_stand. 샘플링 비율과 motion 구간 분리 로깅으로 감시하지 않으면 "항상 standing" 모델도 낮은 CE를 낸다.
3. closed-loop 분포 이동: teacher 궤적 밖 상태에서의 오류 누적. receding horizon과 state 노이즈 증강으로 1차 대응, 부족하면 DAgger식 재라벨(teacher가 closed-loop 재보정 가능하므로 실현 가능).
4. prev_token 입력의 복사 의존: k̂_t ≈ k_{t−1}를 배우면 텍스트를 무시할 수 있다. ablation(§8)으로 확인하고, 심하면 prev_token dropout(학습 중 50% 확률로 k_stand로 대체)을 적용.
5. Teacher 미수렴 샘플(19%, 오차 최대 2.3 cm)이 레이블 노이즈로 들어간다. S0에서는 포함하되 `teacher_converged`로 가중(0.5)하는 옵션을 둔다.
6. 창 길이가 90/94/98 tick으로 다르고 cut 정보는 Student에 없다. 올바른 설계다(cut은 teacher 내부값). 다만 동일 (s₀, d)에서 정지 시점이 다른 샘플이 섞이므로 standing 전이 구간의 레이블이 약간 모호하다.
7. FSQ 32 클래스 중 14개만 관측된다. CE는 문제없지만 unseen 30 cm가 요구하는 토큰이 학습 범위 밖 클래스일 가능성은 낮다(30 cm는 25~35 사이).
8. heading이 ±180°로 다양한 snapshot에서 F0 변환 오류가 나면 held-out bucket 결과를 망친다. `features.py`에 "yaw 회전 불변성" 단위 테스트(같은 궤적을 yaw 회전해도 feature 동일)를 필수로 둔다.
9. Qwen3-0.6B 토크나이저가 한국어 문장을 더 긴 토큰열로 만들며 N이 커진다. key mask로 처리하고 N 상한 32.
10. 평가의 정의: Student가 standing 토큰을 "내서" 멈추는 것과 teacher가 standing으로 "전환"하는 것은 같은 레이블이지만, Student가 멈추지 못하고 계속 걷는 실패(bucket 2, 4에서 가능)를 swing 수와 end speed로 반드시 잡아야 한다.
