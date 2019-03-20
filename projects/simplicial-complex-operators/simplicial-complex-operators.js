"use strict";

/**
 * @module Projects
 */
class SimplicialComplexOperators {

        /** This class implements various operators (e.g. boundary, star, link) on a mesh.
         * @constructor module:Projects.SimplicialComplexOperators
         * @param {module:Core.Mesh} mesh The input mesh this class acts on.
         * @property {module:Core.Mesh} mesh The input mesh this class acts on.
         * @property {module:LinearAlgebra.SparseMatrix} A0 The vertex-edge adjacency matrix of <code>mesh</code>.
         * @property {module:LinearAlgebra.SparseMatrix} A1 The edge-face adjacency matrix of <code>mesh</code>.
         */
        constructor(mesh) {
                this.mesh = mesh;
                this.assignElementIndices(this.mesh);

                this.A0 = this.buildVertexEdgeAdjacencyMatrix(this.mesh);
                this.A1 = this.buildEdgeFaceAdjacencyMatrix(this.mesh);
        }

        /** Assigns indices to the input mesh's vertices, edges, and faces
         * @method module:Projects.SimplicialComplexOperators#assignElementIndices
         * @param {module:Core.Mesh} mesh The input mesh which we index.
         */
        assignElementIndices(mesh) {
                for (let i = 0; i < mesh.vertices.length; i++) {
                        mesh.vertices[i].index = i;
                }
                for (let i = 0; i < mesh.edges.length; i++) {
                        mesh.edges[i].index = i;
                }
                for (let i = 0; i < mesh.faces.length; i++) {
                        mesh.faces[i].index = i;
                }
        }

        /** Returns the vertex-edge adjacency matrix of the given mesh.
         * @method module:Projects.SimplicialComplexOperators#buildVertexEdgeAdjacencyMatrix
         * @param {module:Core.Mesh} mesh The mesh whose adjacency matrix we compute.
         * @returns {module:LinearAlgebra.SparseMatrix} The vertex-edge adjacency matrix of the given mesh.
         */
        buildVertexEdgeAdjacencyMatrix(mesh) {
                let T = new Triplet(mesh.edges.length, mesh.vertices.length);
                for (let i = 0; i < mesh.vertices.length; i++) {
                        for (let e of mesh.vertices[i].adjacentEdges()) {
                                T.addEntry(1, e.index, mesh.vertices[i].index);
                        }
                }
                return SparseMatrix.fromTriplet(T);
        }

        /** Returns the edge-face adjacency matrix.
         * @method module:Projects.SimplicialComplexOperators#buildEdgeFaceAdjacencyMatrix
         * @param {module:Core.Mesh} mesh The mesh whose adjacency matrix we compute.
         * @returns {module:LinearAlgebra.SparseMatrix} The edge-face adjacency matrix of the given mesh.
         */
        buildEdgeFaceAdjacencyMatrix(mesh) {
                let T = new Triplet(mesh.faces.length, mesh.edges.length);
                for (let i = 0; i < mesh.faces.length; i++) {
                        for (let e of mesh.faces[i].adjacentEdges()) {
                                T.addEntry(1, mesh.faces[i].index, e.index);
                        }
                }
                return SparseMatrix.fromTriplet(T);
        }

        /** Returns a column vector representing the vertices of the
         * given subset.
         * @method module:Projects.SimplicialComplexOperators#buildVertexVector
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:LinearAlgebra.DenseMatrix} A column vector with |V| entries. The ith entry is 1 if
         *  vertex i is in the given subset and 0 otherwise
         */
        buildVertexVector(subset) {
                let A = DenseMatrix.zeros(this.mesh.vertices.length, 1);
                for (let v of subset.vertices) {
                        A.set(1, v, 0);
                }
                return A;
        }

        /** Returns a column vector representing the edges of the
         * given subset.
         * @method module:Projects.SimplicialComplexOperators#buildEdgeVector
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:LinearAlgebra.DenseMatrix} A column vector with |E| entries. The ith entry is 1 if
         *  edge i is in the given subset and 0 otherwise
         */
        buildEdgeVector(subset) {
                let A = DenseMatrix.zeros(this.mesh.edges.length, 1);
                for (let e of subset.edges) {
                        A.set(1, e, 0);
                }
                return A;
        }

        /** Returns a column vector representing the faces of the
         * given subset.
         * @method module:Projects.SimplicialComplexOperators#buildFaceVector
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:LinearAlgebra.DenseMatrix} A column vector with |F| entries. The ith entry is 1 if
         *  face i is in the given subset and 0 otherwise
         */
        buildFaceVector(subset) {
                let A = DenseMatrix.zeros(this.mesh.faces.length, 1);
                for (let f of subset.faces) {
                        A.set(1, f, 0);
                }
                return A;
        }

        /** Returns the star of a subset.
         * @method module:Projects.SimplicialComplexOperators#star
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:Core.MeshSubset} The star of the given subset.
         */
        star(subset) {
                let v = this.buildVertexVector(subset);
                let e = this.buildEdgeVector(subset);
                let e1 = this.A0.timesDense(v);
                let f1 = this.A1.timesDense(e);
                let f2 = this.A1.timesDense(e1);
                let star = MeshSubset.deepCopy(subset);
                for (let i = 0; i < this.A0.nRows(); i++) {
                        if (e1.get(i, 0)) {
                                star.addEdge(i);
                        }
                }
                for (let i = 0; i < this.A1.nRows(); i++) {
                        if (f1.get(i, 0) || f2.get(i, 0)) {
                                star.addFace(i);
                        }
                }
                return star;
        }

        /** Returns the closure of a subset.
         * @method module:Projects.SimplicialComplexOperators#closure
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:Core.MeshSubset} The closure of the given subset.
         */
        closure(subset) {
                let e = this.buildEdgeVector(subset);
                let f = this.buildFaceVector(subset);
                let v1 = this.A0.transpose().timesDense(e);
                let e1 = this.A1.transpose().timesDense(f);
                let v2 = this.A0.transpose().timesDense(e1);
                let closure = MeshSubset.deepCopy(subset);
                for (let i = 0; i < this.A1.nCols(); i++) {
                        if (e1.get(i, 0)) {
                                closure.addEdge(i);
                        }
                }
                for (let i = 0; i < this.A0.nCols(); i++) {
                        if (v1.get(i, 0) || v2.get(i, 0)) {
                                closure.addVertex(i);
                        }
                }
                return closure;
        }

        /** Returns the link of a subset.
         * @method module:Projects.SimplicialComplexOperators#link
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:Core.MeshSubset} The link of the given subset.
         */
        link(subset) {
                let link = this.closure(this.star(subset));
                link.deleteSubset(this.star(this.closure(subset)));
                return link;
        }

        /** Returns true if the given subset is a subcomplex and false otherwise.
         * @method module:Projects.SimplicialComplexOperators#isComplex
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {boolean} True if the given subset is a subcomplex and false otherwise.
         */
        isComplex(subset) {
                return subset.equals(this.closure(subset));
        }

        /** Returns the degree if the given subset is a pure subcomplex and -1 otherwise.
         * @method module:Projects.SimplicialComplexOperators#isPureComplex
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {number} The degree of the given subset if it is a pure subcomplex and -1 otherwise.
         */
        isPureComplex(subset) {
                if (!this.isComplex(subset)) {
                        return -1;
                }
                if (subset.faces.size == 0) {
                        if (subset.edges.size == 0) {
                                return 0;
                        } else {
                                for (let v of subset.vertices) {
                                        let star = this.star(new MeshSubset([v], [], []));
                                        let pure = false;
                                        for (let e of star.edges) {
                                                if (subset.edges.has(e)) {
                                                        pure = true;
                                                }
                                        }
                                        if (!pure) return -1;
                                }
                                return 1;
                        }
                } else {
                        for (let v of subset.vertices) {
                                let star = this.star(new MeshSubset([v], [], []));
                                let pure = false;
                                for (let e of star.edges) {
                                        if (subset.edges.has(e)) {
                                                pure = true;
                                        }
                                }
                                if (!pure) return -1;
                        }
                        for (let e of subset.edges) {
                                let star = this.star(new MeshSubset([], [e], []));
                                let pure = false;
                                for (let f of star.faces) {
                                        if (subset.faces.has(f)) {
                                                pure = true;
                                        }
                                }
                                if (!pure) return -1;
                        }
                        return 2;
                }
        }

        /** Returns the boundary of a subset.
         * @method module:Projects.SimplicialComplexOperators#boundary
         * @param {module:Core.MeshSubset} subset A subset of our mesh. We assume <code>subset</code> is a pure subcomplex.
         * @returns {module:Core.MeshSubset} The boundary of the given pure subcomplex.
         */
        boundary(subset) {
                let boundary = new MeshSubset();
                if (this.isPureComplex(subset) == 1) {
                        for (let v of subset.vertices) {
                                let star = this.star(new MeshSubset([v], [], []));
                                let count = 0;
                                for (let e of star.edges) {
                                        if (subset.edges.has(e)) {
                                                count += 1;
                                        }
                                }
                                if (count == 1) {
                                        boundary.addVertex(v);
                                }
                        }
                } else if (this.isPureComplex(subset) == 2) {
                        for (let e of subset.edges) {
                                let star = this.star(new MeshSubset([], [e], []));
                                let count = 0;
                                for (let f of star.faces) {
                                        if (subset.faces.has(f)) {
                                                count += 1;
                                        }
                                }
                                if (count == 1) {
                                        boundary.addEdge(e);
                                }
                        }
                        boundary = this.closure(boundary);
                }
                return boundary;
        }
}
