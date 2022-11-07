<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Code Organization

Header files in include are public APIs that share across modules.
There can be internal header files within each module that sit in src.

## Modules
- support: Internal support utilities.
- runtime: Minimum runtime related codes.
- node: base infra for IR/AST nodes that is dialect independent.
- arith: Arithmetic expression and set simplification.
- top: tensor operation DSL for compute and schedule.
- relay: Implementation of Relay. The second generation of NNVM, a new IR for deep learning frameworks.
- pass: The optimization pass on the IR structure.
- codegen: The code generator.
- autotvm: The auto-tuning module.
- contrib: Contrib extension libraries.
- api: API function registration.
- lang: The definition of DSL related data structure.